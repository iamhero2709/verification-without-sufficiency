"""
modal_stats.py — the two remaining free fixes from the review.

  M4  Bootstrap confidence intervals on every AUC in the paper.
      Stratified over QUESTIONS, not over question-paragraph pairs, because
      paragraphs within a question are correlated. Resampling pairs would give
      intervals that are too narrow.

  M5  Exact McNemar with Holm correction for every end-to-end comparison,
      recomputed from the raw per-question traces so the family structure is
      explicit and consistent.

Both write paste-ready LaTeX into results/tex/, so nothing is hand-typed.

    modal run modal_stats.py --stage decomp   # ~13 min, ~$1  (re-scores and
                                              #  saves per-pair values, which
                                              #  the first run did not keep)
    modal run modal_stats.py --stage auc      # ~6 min,  CPU only
    modal run modal_stats.py --stage tests    # ~2 min,  CPU only
    modal run modal_stats.py --stage all

Then in the paper:  \input{tex/auc_ci.tex}  and  \input{tex/tests.tex}
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "transformers==4.44.2", "tokenizers==0.19.1",
        "sentencepiece==0.2.0", "datasets==2.21.0", "numpy==1.26.4",
        "scipy==1.14.0", "scikit-learn==1.5.1", "pandas==2.2.2",
        "pyarrow==17.0.0", "wandb==0.17.7",
    )
    .add_local_python_source("triver_core")
)

cache = modal.Volume.from_name("triver-cache", create_if_missing=True)
results = modal.Volume.from_name("triver-results", create_if_missing=True)
COMMON = dict(image=image, volumes={"/cache": cache, "/results": results},
              secrets=[modal.Secret.from_name("huggingface-secret"),
                       modal.Secret.from_name("wandb-secret")],
              timeout=60 * 60 * 2, retries=2)

app = modal.App("triver-stats")
R = Path("/results")
NLI_ID = "cross-encoder/nli-deberta-v3-base"
N_BOOT = 1000
SEED = 1337

_C = {}


def _env():
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/cache/hf/datasets")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ============================================================================
# Fast AUC and question-stratified bootstrap
# ============================================================================

def fast_auc(y, s):
    """Mann-Whitney U form. Handles ties through average ranks, which matters
    here because entailment scores pile up near zero."""
    import numpy as np
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_auc(y, s, qids, n_boot=N_BOOT, seed=SEED):
    """Resample QUESTIONS with replacement, keeping each question's paragraphs
    together. Resampling paragraphs independently would understate the
    interval, since paragraphs from one question share a query embedding."""
    import numpy as np
    y = np.asarray(y); s = np.asarray(s, dtype=float); q = np.asarray(qids)
    point = fast_auc(y, s)
    if not np.isfinite(point):
        return point, float("nan"), float("nan")
    uq = np.unique(q)
    idx_by_q = {u: np.flatnonzero(q == u) for u in uq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([idx_by_q[u] for u in pick])
        a = fast_auc(y[idx], s[idx])
        if np.isfinite(a):
            boots.append(a)
    if len(boots) < 50:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def fmt(v, lo, hi, bold=False):
    import numpy as np
    if not np.isfinite(v):
        return "n/a"
    core = f"{v:.3f}"
    if bold:
        core = r"\textbf{" + core + "}"
    if not np.isfinite(lo):
        return core
    return core + r"\,\tiny{[" + f"{lo:.3f}, {hi:.3f}" + r"]}"


# ============================================================================
# Stage: re-score the decomposition experiment, saving per-pair values
# ============================================================================

def nli(pairs, batch=32, max_len=512):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if "nli" not in _C:
        _env()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(NLI_ID)
        mdl = AutoModelForSequenceClassification.from_pretrained(NLI_ID).to(dev).eval()
        id2 = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
        _C["nli"] = (tok, mdl, dev, next(i for i, v in id2.items() if v == "entailment"))
        print(f"[load] {NLI_ID}")
    tok, mdl, dev, ix = _C["nli"]
    out = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch):
            b = pairs[i:i + batch]
            enc = tok([p for p, _ in b], [h for _, h in b], return_tensors="pt",
                      padding=True, truncation=True, max_length=max_len).to(dev)
            out.extend(torch.softmax(mdl(**enc).logits, -1)[:, ix].float().cpu().tolist())
    return out


@app.function(gpu="T4", cpu=4, memory=16384, **COMMON)
def decomp_rescore() -> dict:
    """The first decomposition run saved only aggregates. The paper's headline
    positive result needs an interval, so we re-score and keep every value."""
    _env()
    from datasets import load_dataset
    import triver_core as tc

    out_path = R / "stats" / "decomp_pairs.jsonl"
    if out_path.exists():
        print("already scored"); return {"cached": True}

    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    keep = {r["qid"] for r in read_jsonl(R / "adv" / "splits" / "musique.jsonl")}
    ex_by_id = {str(e["id"]): e for e in ds if str(e["id"]) in keep}

    def resolve(t, prev):
        for k, a in prev.items():
            t = t.replace(f"#{k}", str(a))
        return t

    meta, p_orig, p_sub, p_sub_or = [], [], [], []
    for qid, ex in ex_by_id.items():
        paras = ex["paragraphs"]
        decomp = ex.get("question_decomposition") or []
        if len(decomp) < 2:
            continue
        prev = {}
        for step_i, step in enumerate(decomp, start=1):
            sub = resolve(str(step.get("question", "")), prev)
            prev[step_i] = step.get("answer", "")
            idx = step.get("paragraph_support_idx")
            if idx is None or step_i == 1:
                continue
            subj, rel = ([x.strip() for x in sub.split(">>", 1)]
                         if ">>" in sub else (sub, "answer"))
            h_sub = f"The {rel} of {subj} is something."
            h_or = f"The {rel} of {subj} is {step.get('answer','')}."
            h_q = tc.build_hypotheses(ex["question"], "prop_slot")[0]
            for j, p in enumerate(paras):
                t = (str(p.get("title", "")) + ". " +
                     str(p.get("paragraph_text", ""))).strip()
                p_orig.append((t, h_q)); p_sub.append((t, h_sub))
                p_sub_or.append((t, h_or))
                meta.append({"qid": qid, "n_hops": len(decomp),
                             "label": int(int(p.get("idx", j)) == int(idx))})
            break

    print(f"scoring {len(p_orig)} pairs x 3 forms")
    s_o, s_s, s_a = nli(p_orig), nli(p_sub), nli(p_sub_or)
    rows = [{**m, "s_original": a, "s_decomposed": b, "s_decomposed_answer": c}
            for m, a, b, c in zip(meta, s_o, s_s, s_a)]
    write_jsonl(out_path, rows)
    results.commit()
    print(f"saved {len(rows)} rows")
    return {"n": len(rows)}


# ============================================================================
# Stage: AUC with intervals
# ============================================================================

@app.function(cpu=16, memory=32768, **COMMON)
def auc_intervals() -> dict:
    _env()
    import numpy as np
    import pandas as pd
    import triver_core as tc

    TEX = R / "results" / "tex"; TEX.mkdir(parents=True, exist_ok=True)
    AN = R / "results" / "analysis"; AN.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---------- main signal table, by dataset and evidence role
    sig = pd.read_parquet(R / "adv" / "signals.parquet")
    text_of, ans_of = {}, {}
    for ds in ("hotpotqa", "2wiki", "musique"):
        f = R / "adv" / "splits" / f"{ds}.jsonl"
        if f.exists():
            for rec in read_jsonl(f):
                for p in rec["paragraphs"]:
                    text_of[p["pid"]] = p["text"]; ans_of[p["pid"]] = rec["answer"]
    sig["text"] = sig.pid.map(text_of); sig["answer"] = sig.pid.map(ans_of)
    sig = sig.dropna(subset=["text", "answer"])
    if "answer_bearing" not in sig.columns:
        sig["answer_bearing"] = sig.apply(
            lambda r: tc.normalize_answer(str(r.answer)) not in ("yes", "no", "")
            and tc.normalize_answer(str(r.answer)) in tc.normalize_answer(str(r.text)),
            axis=1)

    signals = ["s_emb", "s_ent_oracle", "s_ent_slot"]
    for ds, d in sig.groupby("dataset"):
        dis = d[~d.is_gold]
        for c in signals:
            for split, pos in (("all gold", d[d.is_gold]),
                               ("answer-bearing", d[d.is_gold & d.answer_bearing]),
                               ("bridge-only", d[d.is_gold & ~d.answer_bearing])):
                if len(pos) < 20:
                    continue
                y = np.r_[np.ones(len(pos)), np.zeros(len(dis))]
                s = np.r_[pos[c].values, dis[c].values]
                q = np.r_[pos.qid.values, dis.qid.values]
                v, lo, hi = bootstrap_auc(y, s, q)
                rows.append({"table": "signals", "dataset": ds, "signal": c,
                             "split": split, "auc": v, "lo": lo, "hi": hi,
                             "n_pos": len(pos), "n_q": int(d.qid.nunique())})
        print(f"  {ds} done")

    # ---------- hop scaling on MuSiQue
    mq = sig[sig.dataset == "musique"]
    for h, d in mq.groupby("n_hops"):
        dis = d[~d.is_gold]; pos = d[d.is_gold]
        for c in signals:
            if len(pos) < 20:
                continue
            y = np.r_[np.ones(len(pos)), np.zeros(len(dis))]
            s = np.r_[pos[c].values, dis[c].values]
            q = np.r_[pos.qid.values, dis.qid.values]
            v, lo, hi = bootstrap_auc(y, s, q)
            rows.append({"table": "hops", "dataset": "musique", "signal": c,
                         "split": f"{int(h)}hop", "auc": v, "lo": lo, "hi": hi,
                         "n_pos": len(pos), "n_q": int(d.qid.nunique())})

    # ---------- decomposition, the headline positive result
    # prefer the corrected scoring from modal_llmdecomp, which resolves the
    # #k placeholders and builds object-level propositions from natural-language
    # sub-questions instead of forcing every one into a relational template
    dp = R / "llmdecomp" / "scored_v2.jsonl"
    use_v2 = dp.exists()
    if not use_v2:
        dp = R / "stats" / "decomp_pairs.jsonl"
    if dp.exists():
        dd = pd.DataFrame(read_jsonl(dp))
        print(f"  decomposition source: {dp.name}")
        if use_v2:
            dd = dd.rename(columns={"s_gold": "s_decomposed",
                                    "s_anchored": "s_llm_anchored",
                                    "s_blind": "s_llm_blind"})
            dd["s_original"] = float("nan")
        cols = [("s_original", "original question"),
                ("s_decomposed", "decomposed sub-question"),
                ("s_decomposed_answer", "sub-question with answer")]
        if use_v2:
            cols = [("s_llm_blind", "LLM, question only"),
                    ("s_llm_anchored", "LLM, anchored on top-1 chunk"),
                    ("s_decomposed", "gold sub-question")]
        for col, name in cols:
            if col not in dd.columns or dd[col].isna().all():
                continue
            v, lo, hi = bootstrap_auc(dd.label.values, dd[col].values, dd.qid.values)
            rows.append({"table": "decomp", "dataset": "musique", "signal": col,
                         "split": name, "auc": v, "lo": lo, "hi": hi,
                         "n_pos": int(dd.label.sum()), "n_q": int(dd.qid.nunique())})
        hop_cols = (("s_original", "s_decomposed", "s_llm_anchored")
                    if use_v2 else ("s_original", "s_decomposed"))
        for h, d in dd.groupby("n_hops"):
            for col in hop_cols:
                if col not in d.columns or d[col].isna().all():
                    continue
                v, lo, hi = bootstrap_auc(d.label.values, d[col].values, d.qid.values)
                rows.append({"table": "decomp_hops", "dataset": "musique",
                             "signal": col, "split": f"{int(h)}hop",
                             "auc": v, "lo": lo, "hi": hi,
                             "n_pos": int(d.label.sum()),
                             "n_q": int(d.qid.nunique())})
        # paired bootstrap on the difference, which is what the claim needs.
        # When reading scored_v2 the original-question scores live in the older
        # file, so join them in on qid order.
        if use_v2:
            old = R / "stats" / "decomp_pairs.jsonl"
            if old.exists():
                od = pd.DataFrame(read_jsonl(old))
                if len(od) == len(dd):
                    dd["s_original"] = od["s_original"].values
        rng = np.random.default_rng(SEED)
        uq = dd.qid.unique()
        idx_by_q = {u: np.flatnonzero(dd.qid.values == u) for u in uq}
        diffs = []
        for _ in range(N_BOOT):
            pick = rng.choice(uq, len(uq), replace=True)
            idx = np.concatenate([idx_by_q[u] for u in pick])
            a = fast_auc(dd.label.values[idx], dd.s_original.values[idx])
            b = fast_auc(dd.label.values[idx], dd.s_decomposed.values[idx])
            if np.isfinite(a) and np.isfinite(b):
                diffs.append(b - a)
        dlo, dhi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"table": "decomp", "dataset": "musique",
                     "signal": "lift_decomposed_minus_original", "split": "paired",
                     "auc": float(np.mean(diffs)), "lo": float(dlo), "hi": float(dhi),
                     "n_pos": int(dd.label.sum()), "n_q": len(uq)})
        print(f"  decomposition lift {np.mean(diffs):+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
    else:
        print("  decomp_pairs.jsonl missing; run --stage decomp first")

    t = pd.DataFrame(rows)
    t.to_csv(AN / "auc_with_ci.csv", index=False)

    # ---------- emit LaTeX
    def block(sub, caption, label):
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{" + caption + r"}", r"\label{" + label + r"}",
                 r"\setlength{\tabcolsep}{4pt}",
                 r"\begin{tabular}{@{}llc@{}}", r"\toprule",
                 r"\textbf{Dataset} & \textbf{Signal / split} & "
                 r"\textbf{AUC \tiny{[95\% CI]}}\\", r"\midrule"]
        for _, r_ in sub.iterrows():
            lines.append(f"{r_.dataset} & {r_.signal} ({r_.split}) & "
                         + fmt(r_.auc, r_.lo, r_.hi) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        return "\n".join(lines)

    tex = ["% generated by modal_stats.py, do not edit by hand",
           "% intervals are question-stratified bootstrap, "
           f"{N_BOOT} resamples, seed {SEED}", ""]
    for key, cap, lab in (
        ("signals", "Separating gold paragraphs from distractors, with "
                    "question-stratified bootstrap intervals.", "tab:auc-ci"),
        ("hops", "AUC by hop count on MuSiQue, with intervals.", "tab:hops-ci"),
        ("decomp", "Identifying the paragraph supporting a later hop, with "
                   "intervals. The final row is a paired bootstrap on the "
                   "difference.", "tab:decomp-ci"),
        ("decomp_hops", "Decomposition against the original question, by hop "
                        "count, with intervals.", "tab:decomp-hops-ci"),
    ):
        sub = t[t.table == key]
        if len(sub):
            tex.append(block(sub, cap, lab))
    (TEX / "auc_ci.tex").write_text("\n".join(tex))
    results.commit()

    print("\n" + t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {TEX/'auc_ci.tex'}")
    return {"n_rows": len(t)}


# ============================================================================
# Stage: paired significance tests
# ============================================================================

@app.function(cpu=16, memory=32768, **COMMON)
def paired_tests(reference: str = "B1") -> dict:
    """Recompute every end-to-end comparison from the raw traces against a
    single reference, so the family structure for Holm is explicit. B1, passing
    all retrieved chunks unfiltered, is the right reference because the paper's
    claim is about whether filtering helps at all."""
    import numpy as np
    import pandas as pd
    import triver_core as tc

    TEX = R / "results" / "tex"; TEX.mkdir(parents=True, exist_ok=True)
    AN = R / "results" / "analysis"; AN.mkdir(parents=True, exist_ok=True)

    cells = {}          # (prompt, dataset, generator) -> {config: DataFrame}
    for root, prompt in ((R / "results" / "adv_gen", "gen_v3"),
                         (R / "results" / "v2" / "gen", None)):
        if not root.exists():
            continue
        if prompt is None:
            iters = [(p.name, p) for p in sorted(root.glob("*"))]
        else:
            iters = [(prompt, root)]
        for pname, proot in iters:
            for dsdir in sorted(proot.glob("*")):
                for mdir in sorted(dsdir.glob("*")):
                    for cdir in sorted(mdir.glob("*")):
                        rr = [r for f in sorted(cdir.glob("*.jsonl"))
                              for r in read_jsonl(f)]
                        if not rr:
                            continue
                        key = (pname, dsdir.name, mdir.name)
                        cells.setdefault(key, {})[cdir.name] = \
                            pd.DataFrame(rr).set_index("qid").sort_index()

    rows = []
    for (pname, ds, gen), per in sorted(cells.items()):
        if reference not in per:
            continue
        ref = per[reference]
        fam = []
        for cfg, d in sorted(per.items()):
            if cfg == reference:
                continue
            idx = ref.index.intersection(d.index)
            if len(idx) < 30:
                continue
            p, b, c = tc.mcnemar_exact(ref.loc[idx].em, d.loc[idx].em)
            flo, fhi = tc.paired_bootstrap(ref.loc[idx].f1, d.loc[idx].f1)
            fam.append({"prompt": pname, "dataset": ds, "generator": gen,
                        "config": cfg, "vs": reference, "n": len(idx),
                        "b": b, "c": c, "p_raw": p,
                        "em_ref": 100 * ref.loc[idx].em.mean(),
                        "em_cfg": 100 * d.loc[idx].em.mean(),
                        "em_delta": 100 * (d.loc[idx].em.mean()
                                           - ref.loc[idx].em.mean()),
                        "f1_delta_lo": flo, "f1_delta_hi": fhi})
        if fam:
            for r_, adj in zip(fam, tc.holm([x["p_raw"] for x in fam])):
                r_["p_holm"] = adj
                r_["sig"] = adj < 0.05
            rows.extend(fam)

    t = pd.DataFrame(rows)
    t.to_csv(AN / "paired_tests_vs_B1.csv", index=False)

    def stars(p):
        return r"$^{***}$" if p < .001 else r"$^{**}$" if p < .01 \
            else r"$^{*}$" if p < .05 else ""

    main = t[(t.prompt == "gen_v3") & (t.generator == "Qwen2.5-1.5B-Instruct")]
    lines = ["% generated by modal_stats.py, do not edit by hand",
             r"\begin{table}[t]", r"\centering",
             r"\caption{Paired exact McNemar against B1, unfiltered retrieval. "
             r"$b$ and $c$ are discordant counts. $p$ is Holm-adjusted within "
             r"each dataset. Qwen2.5-1.5B, standard prompt. "
             r"$^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$.}",
             r"\label{tab:tests}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Selector} & $\Delta$\textbf{EM} & "
             r"$b$ & $c$ & \textbf{Holm }$p$\\", r"\midrule"]
    for ds in sorted(main.dataset.unique()):
        sub = main[main.dataset == ds].sort_values("em_delta", ascending=False)
        for i, (_, r_) in enumerate(sub.iterrows()):
            lines.append(
                f"{ds if i == 0 else ''} & {r_.config} & "
                f"{r_.em_delta:+.1f} & {int(r_.b)} & {int(r_.c)} & "
                f"{r_.p_holm:.3f}{stars(r_.p_holm)}" + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    (TEX / "tests.tex").write_text("\n".join(lines))
    results.commit()

    print("\n" + t[["prompt", "dataset", "generator", "config", "em_delta",
                    "b", "c", "p_raw", "p_holm", "sig"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n--- what survives Holm at 0.05, standard prompt, 1.5B ---")
    for _, r_ in main[main.sig].iterrows():
        print(f"  {r_.dataset:9s} {r_.config:5s} vs B1: "
              f"{r_.em_delta:+.1f} EM, p_holm={r_.p_holm:.4f}")
    if not len(main[main.sig]):
        print("  none")
    print(f"\nwrote {TEX/'tests.tex'}")
    return {"n_tests": len(t), "n_significant": int(t.sig.sum())}


# ============================================================================

@app.local_entrypoint()
def main(stage: str = "all"):
    t0 = time.time()
    if stage in ("decomp", "all"):
        print(">> re-scoring the decomposition experiment with per-pair output")
        print(decomp_rescore.remote())
    if stage in ("auc", "all"):
        print(">> bootstrap intervals on every AUC")
        auc_intervals.remote()
    if stage in ("tests", "all"):
        print(">> paired McNemar with Holm, against B1")
        print(paired_tests.remote())
    print(f"\nwall clock {(time.time()-t0)/60:.1f} min")
    print("pull the fragments:  modal volume get triver-results "
          "/results/tex ./tex")
