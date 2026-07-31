"""
modal_setlevel.py — test the fix the paper proposes.

D5 showed set-level entailment reaches 0.881 AUC against 0.664 per chunk, but it
used the ORACLE hypothesis, which contains the gold answer and cannot be
deployed. Before spending anything on generation we measure the same thing with
the SLOT hypothesis, which is what a real system would use.

    modal run modal_setlevel.py --stage pairs     # ~4 min,  ~$0.20   <- gate
    modal run modal_setlevel.py --stage gen       # ~50 min, ~$3
    modal run modal_setlevel.py --stage analyze   # free

Stage `pairs` prints the number that decides whether stage `gen` is worth
running. If set-level SLOT AUC is not well above the per-chunk SLOT AUC of
0.620, the proposal is weaker than D5 suggested and you should say so in the
paper rather than build on it.

Configurations in stage `gen`:
  B1   top-5, no verification
  B2   top-2 by retriever score          (context-length-matched control)
  B4   cross-encoder reranker, top-2     (the standard baseline)
  PC   per-chunk NLI gate                (what the paper says fails)
  SET  best pair by set-level NLI        (what the paper proposes)
  B5   oracle gold pair                  (upper bound)
"""
from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "transformers==4.44.2", "tokenizers==0.19.1",
        "sentencepiece==0.2.0", "sentence-transformers==3.0.1",
        "accelerate==0.33.0", "numpy==1.26.4", "scipy==1.14.0",
        "scikit-learn==1.5.1", "pandas==2.2.2", "pyarrow==17.0.0",
        "matplotlib==3.9.2", "wandb==0.17.7",
    )
    .add_local_python_source("triver_core")
)

cache = modal.Volume.from_name("triver-cache", create_if_missing=True)
results = modal.Volume.from_name("triver-results", create_if_missing=True)
COMMON = dict(image=image, volumes={"/cache": cache, "/results": results},
              secrets=[modal.Secret.from_name("huggingface-secret"),
                       modal.Secret.from_name("wandb-secret")],
              timeout=60 * 60 * 2, retries=2)

app = modal.App("triver-setlevel")
R = Path("/results")

NLI_ID = "cross-encoder/nli-deberta-v3-base"
RERANK_ID = "BAAI/bge-reranker-base"
GEN_IDS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
TOP_K = 5
SEED = 1337
CONFIGS = ["B1", "B2", "B4", "PC", "SET", "B5"]
PC_TAU = 0.35          # per-chunk gate; sensitivity is reported in analyze

_C = {}


def _env():
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def seed_all():
    import random
    import numpy as np
    import torch
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


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
        print(f"[load] {NLI_ID} on {dev}")
    tok, mdl, dev, ix = _C["nli"]
    out = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch):
            b = pairs[i:i + batch]
            enc = tok([p for p, _ in b], [h for _, h in b], return_tensors="pt",
                      padding=True, truncation=True, max_length=max_len).to(dev)
            out.extend(torch.softmax(mdl(**enc).logits, -1)[:, ix].float().cpu().tolist())
    return out


def rerank(query, texts):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if "rr" not in _C:
        _env()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(RERANK_ID)
        mdl = AutoModelForSequenceClassification.from_pretrained(RERANK_ID).to(dev).eval()
        _C["rr"] = (tok, mdl, dev)
        print(f"[load] {RERANK_ID} on {dev}")
    tok, mdl, dev = _C["rr"]
    with torch.inference_mode():
        enc = tok([query] * len(texts), texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(dev)
        return mdl(**enc).logits.view(-1).float().cpu().tolist()


def generate(model_id, prompt, max_new_tokens=32):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    key = f"gen::{model_id}"
    if key not in _C:
        _env()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.bfloat16 if dev == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt).to(dev).eval()
        _C[key] = (tok, mdl, dev)
        print(f"[load] {model_id} on {dev}")
    tok, mdl, dev = _C[key]
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok([text], return_tensors="pt").to(dev)
    with torch.inference_mode():
        o = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         num_beams=1, pad_token_id=tok.eos_token_id)
    n_in = int(enc["input_ids"].shape[1])
    return tok.decode(o[0][n_in:], skip_special_tokens=True).strip(), n_in


# ============================================================================
# Stage 1 — pair scores, and the gate that decides the rest
# ============================================================================

@app.function(gpu="T4", cpu=4, memory=16384, **COMMON)
def pair_shard(lo: int, hi: int) -> str:
    """Score all C(TOP_K,2) pairs of retrieved chunks, both hypothesis modes."""
    _env(); seed_all()
    import pandas as pd
    import triver_core as tc

    out_path = R / "artifacts" / "pairs" / f"{lo:05d}.jsonl"
    if out_path.exists():
        return str(out_path)

    sig = pd.read_parquet(R / "artifacts" / "signals.parquet").set_index("pid").to_dict("index")
    recs = read_jsonl(R / "splits" / "eval.jsonl")[lo:hi]

    rows = []
    for rec in recs:
        q, gold = rec["question"], rec["answer"]
        h_slot = tc.build_hypotheses(q, "prop_slot")[0]
        h_orac = tc.build_hypotheses(q, "prop_oracle", gold=gold)[0]
        top = sorted(rec["paragraphs"],
                     key=lambda p: -sig[p["pid"]]["retriever_score"])[:TOP_K]
        combos = list(itertools.combinations(range(len(top)), 2))
        prem = [top[a]["text"] + " " + top[b]["text"] for a, b in combos]
        s_slot = nli([(t, h_slot) for t in prem])
        s_orac = nli([(t, h_orac) for t in prem])
        for (a, b), ss, so in zip(combos, s_slot, s_orac):
            rows.append({
                "qid": rec["qid"], "pid_a": top[a]["pid"], "pid_b": top[b]["pid"],
                "both_gold": bool(top[a]["is_gold"] and top[b]["is_gold"]),
                "n_gold": int(top[a]["is_gold"]) + int(top[b]["is_gold"]),
                "s_pair_slot": ss, "s_pair_oracle": so,
            })
    write_jsonl(out_path, rows)
    results.commit()
    return str(out_path)


@app.function(cpu=8, memory=16384, **COMMON)
def pair_auc() -> dict:
    """THE GATE. Is the deployable set-level signal actually strong?"""
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    files = sorted((R / "artifacts" / "pairs").glob("*.jsonl"))
    df = pd.DataFrame([r for f in files for r in read_jsonl(f)])
    df.to_parquet(R / "artifacts" / "pairs.parquet")

    out = {
        "n_questions": int(df.qid.nunique()), "n_pairs": len(df),
        "pairs_with_both_gold": int(df.both_gold.sum()),
        "AUC_pair_slot": round(float(roc_auc_score(df.both_gold, df.s_pair_slot)), 4),
        "AUC_pair_oracle": round(float(roc_auc_score(df.both_gold, df.s_pair_oracle)), 4),
        "perchunk_slot_reference": 0.620,
        "perchunk_oracle_reference": 0.664,
        "mean_slot_both_gold": round(float(df[df.both_gold].s_pair_slot.mean()), 4),
        "mean_slot_one_gold": round(float(df[df.n_gold == 1].s_pair_slot.mean()), 4),
        "mean_slot_no_gold": round(float(df[df.n_gold == 0].s_pair_slot.mean()), 4),
    }
    out["slot_lift_over_perchunk"] = round(out["AUC_pair_slot"] - 0.620, 4)

    # how often does argmax over pairs land on the true gold pair?
    top1 = df.loc[df.groupby("qid").s_pair_slot.idxmax()]
    out["argmax_pair_is_gold_pair"] = round(float(top1.both_gold.mean()), 4)
    out["argmax_pair_contains_a_gold"] = round(float((top1.n_gold >= 1).mean()), 4)

    d = R / "results" / "analysis"; d.mkdir(parents=True, exist_ok=True)
    (d / "setlevel_pair_auc.json").write_text(json.dumps(out, indent=2))
    results.commit()

    print("\n" + json.dumps(out, indent=2))
    print("\n" + "=" * 70)
    if out["AUC_pair_slot"] >= 0.78:
        print(f"  DEPLOYABLE. Set-level SLOT reaches {out['AUC_pair_slot']:.3f} against")
        print(f"  {0.620:.3f} per chunk, a lift of {out['slot_lift_over_perchunk']:+.3f}.")
        print("  Run --stage gen.")
    elif out["AUC_pair_slot"] >= 0.70:
        print(f"  PARTIAL. {out['AUC_pair_slot']:.3f} beats per-chunk but trails the")
        print(f"  oracle at {out['AUC_pair_oracle']:.3f}. Worth running gen, and the")
        print("  paper must report both numbers so the gap is visible.")
    else:
        print(f"  WEAK. {out['AUC_pair_slot']:.3f} is close to per-chunk. The set-level")
        print("  result depends on seeing the answer, so it is a diagnostic and")
        print("  not a proposal. Say that in the paper and skip generation.")
    print("=" * 70)

    with __import__("wandb").init(project="triver-rag", job_type="setlevel_gate",
                                  name="setlevel_gate", group="paper-v1", reinit=True):
        import wandb
        wandb.log(out)
    return out


# ============================================================================
# Stage 2 — end-to-end generation
# ============================================================================

def select(cfg, rec, sig, pairs_for_q):
    """Return the retained paragraphs for one configuration."""
    top = sorted(rec["paragraphs"], key=lambda p: -sig[p["pid"]]["retriever_score"])[:TOP_K]
    by_pid = {p["pid"]: p for p in rec["paragraphs"]}

    if cfg == "B1":
        return top
    if cfg == "B2":
        return top[:2]
    if cfg == "B5":
        return [p for p in rec["paragraphs"] if p["is_gold"]]
    if cfg == "B4":
        sc = rerank(rec["question"], [p["text"] for p in top])
        return [p for _, p in sorted(zip(sc, top), key=lambda x: -x[0])][:2]
    if cfg == "PC":
        import triver_core as tc
        h = tc.build_hypotheses(rec["question"], "prop_slot")[0]
        sc = nli([(p["text"], h) for p in top])
        keep = [p for p, s in zip(top, sc) if s >= PC_TAU]
        return keep if keep else top[:1]
    if cfg == "SET":
        best = max(pairs_for_q, key=lambda r: r["s_pair_slot"])
        return [by_pid[best["pid_a"]], by_pid[best["pid_b"]]]
    raise ValueError(cfg)


@app.function(gpu="L4", cpu=4, memory=32768, **COMMON)
def gen_shard(cfg: str, lo: int, hi: int, model_id: str) -> str:
    _env(); seed_all()
    import pandas as pd
    import triver_core as tc

    tag = model_id.split("/")[-1]
    out_path = R / "results" / "setlevel" / tag / cfg / f"{lo:05d}.jsonl"
    if out_path.exists():
        return str(out_path)

    sig = pd.read_parquet(R / "artifacts" / "signals.parquet").set_index("pid").to_dict("index")
    pdf = pd.read_parquet(R / "artifacts" / "pairs.parquet")
    pairs_by_q = {k: v.to_dict("records") for k, v in pdf.groupby("qid")}
    recs = read_jsonl(R / "splits" / "eval.jsonl")[lo:hi]

    rows = []
    for rec in recs:
        kept = select(cfg, rec, sig, pairs_by_q.get(rec["qid"], []))
        ctx = "\n\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(kept))
        prompt = tc.PROMPTS["gen_v3"].format(context=ctx or "(no context)",
                                             question=rec["question"])
        t0 = time.perf_counter()
        pred, n_in = generate(model_id, prompt)
        rows.append({
            "qid": rec["qid"], "config": cfg, "generator": model_id,
            "n_kept": len(kept), "context_tokens": n_in,
            "gold_kept": sum(1 for p in kept if p["is_gold"]),
            "gold_available": sum(1 for p in rec["paragraphs"] if p["is_gold"]),
            "pred": pred, "gold": rec["answer"],
            "em": tc.exact_match(pred, rec["answer"]),
            "f1": tc.token_f1(pred, rec["answer"]),
            "t_gen": time.perf_counter() - t0,
        })
    write_jsonl(out_path, rows)
    results.commit()
    return str(out_path)


# ============================================================================
# Stage 3 — analysis
# ============================================================================

@app.function(cpu=8, memory=32768, **COMMON)
def analyze_sets() -> dict:
    import numpy as np
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import triver_core as tc

    AN = R / "results" / "analysis"; AN.mkdir(parents=True, exist_ok=True)
    FIG = R / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

    all_rows, tables = [], []
    for mdir in sorted((R / "results" / "setlevel").glob("*")):
        per = {}
        for cdir in sorted(mdir.glob("*")):
            rows = [r for f in sorted(cdir.glob("*.jsonl")) for r in read_jsonl(f)]
            if rows:
                per[cdir.name] = pd.DataFrame(rows).set_index("qid").sort_index()
                all_rows.extend(rows)
        for name, d in per.items():
            lo, hi = tc.wilson(int(d.em.sum()), len(d))
            tables.append({
                "generator": mdir.name, "config": name, "n": len(d),
                "EM": 100 * d.em.mean(), "CI_lo": 100 * lo, "CI_hi": 100 * hi,
                "F1": d.f1.mean(), "n_kept": d.n_kept.mean(),
                "ctx_tokens": d.context_tokens.mean(),
                "gold_recall": (d.gold_kept / d.gold_available.clip(lower=1)).mean(),
            })
        # paired tests against B2 within this generator
        if "B2" in per:
            ref, tests = per["B2"], []
            for name, d in per.items():
                if name == "B2":
                    continue
                idx = ref.index.intersection(d.index)
                p, b, c = tc.mcnemar_exact(ref.loc[idx].em, d.loc[idx].em)
                tests.append({"generator": mdir.name, "config": name, "vs": "B2",
                              "n": len(idx), "b": b, "c": c, "p_raw": p,
                              "em_delta": 100 * (d.loc[idx].em.mean() - ref.loc[idx].em.mean())})
            if tests:
                for t, a in zip(tests, tc.holm([x["p_raw"] for x in tests])):
                    t["p_holm"] = a
                pd.DataFrame(tests).to_csv(
                    AN / f"setlevel_tests_{mdir.name}.csv", index=False)

    mt = pd.DataFrame(tables).sort_values(["generator", "EM"], ascending=[True, False])
    mt.to_csv(AN / "setlevel_main_table.csv", index=False)
    print("\n" + mt.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    order = ["B1", "B2", "B4", "PC", "SET", "B5"]
    gens = sorted(mt.generator.unique())
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.8 / max(len(gens), 1)
    for i, g in enumerate(gens):
        d = mt[mt.generator == g].set_index("config").reindex(order)
        x = np.arange(len(order)) + i * w
        err = np.vstack([d.EM - d.CI_lo, d.CI_hi - d.EM])
        ax.bar(x, d.EM, w * 0.9, yerr=err, capsize=3, label=g,
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(order)) + w * (len(gens) - 1) / 2)
    ax.set_xticklabels(order)
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("Per-chunk (PC) vs set-level (SET) verification, 95% Wilson intervals")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_setlevel.pdf"); fig.savefig(FIG / "fig_setlevel.png", dpi=160)
    results.commit()

    with __import__("wandb").init(project="triver-rag", job_type="setlevel_analyze",
                                  name="setlevel_analyze", group="paper-v1", reinit=True):
        import wandb
        wandb.log({"setlevel_table": wandb.Table(dataframe=mt),
                   "fig_setlevel": wandb.Image(str(FIG / "fig_setlevel.png"))})
    return {"table": mt.to_dict("records")}


# ============================================================================

def _shards(n, size):
    return [(i, min(i + size, n)) for i in range(0, n, size)]


@app.local_entrypoint()
def main(stage: str = "pairs", n_eval: int = 500, shard: int = 50):
    t0 = time.time()

    if stage in ("pairs", "all"):
        jobs = _shards(n_eval, shard)
        print(f">> pair scoring: {len(jobs)} shards")
        list(pair_shard.starmap(jobs))
        gate = pair_auc.remote()
        if gate["AUC_pair_slot"] < 0.70 and stage == "all":
            print("\nDeployable set-level signal is weak. Stopping before generation.")
            return

    if stage in ("gen", "all"):
        jobs = [(c, lo, hi, m) for m in GEN_IDS for c in CONFIGS
                for lo, hi in _shards(n_eval, shard)]
        print(f">> generation: {len(jobs)} shards, {len(CONFIGS)} configs x {len(GEN_IDS)} models")
        list(gen_shard.starmap(jobs))

    if stage in ("gen", "analyze", "all"):
        print(">> analyze")
        analyze_sets.remote()

    print(f"\nwall clock {(time.time()-t0)/60:.1f} min")
