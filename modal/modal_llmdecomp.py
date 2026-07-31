"""
modal_llmdecomp.py — turn the decomposition ceiling into a deployable number.

The paper's positive result uses MuSiQue's gold `question_decomposition`, so it
is a ceiling and not a method. Both reviews flagged the same gap. This closes it
by having an off-the-shelf LLM produce the decomposition and re-measuring.

Three decomposer settings, all with no gold annotation at inference time:

  blind     The LLM sees only the question and is asked what must be answered
            first, then what follows. Pure zero-shot decomposition.

  anchored  The LLM sees the question and the top-1 retrieved paragraph, and is
            asked for the follow-up question. This is the Self-Ask shape and it
            is what a real pipeline would do, since retrieval has already run.

  gold      MuSiQue's annotation, for reference. Already measured at 0.779.

The number that matters is whether `anchored` lands meaningfully above the
original-question baseline of 0.546. If it reaches, say, 0.68, the paper stops
saying "a ceiling exists" and starts saying "here is a method, and here is how
much of the ceiling it reaches."

    modal run modal_llmdecomp.py --stage decompose   # ~35 min, ~$1
    modal run modal_llmdecomp.py --stage score       # ~5 min,  ~$0.3
    modal run modal_llmdecomp.py --stage analyze     # free

Requires artefacts from modal_advanced.py and modal_stats.py.
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
        "sentencepiece==0.2.0", "sentence-transformers==3.0.1",
        "accelerate==0.33.0", "datasets==2.21.0", "numpy==1.26.4",
        "scipy==1.14.0", "scikit-learn==1.5.1", "pandas==2.2.2",
        "pyarrow==17.0.0", "matplotlib==3.9.2", "wandb==0.17.7",
    )
    .add_local_python_source("triver_core")
)

cache = modal.Volume.from_name("triver-cache", create_if_missing=True)
results = modal.Volume.from_name("triver-results", create_if_missing=True)
COMMON = dict(image=image, volumes={"/cache": cache, "/results": results},
              secrets=[modal.Secret.from_name("huggingface-secret"),
                       modal.Secret.from_name("wandb-secret")],
              timeout=60 * 60 * 2, retries=2)

app = modal.App("triver-llmdecomp")
R = Path("/results")

DECOMPOSER = "Qwen/Qwen2.5-7B-Instruct"   # off-the-shelf, not fine-tuned
NLI_ID = "cross-encoder/nli-deberta-v3-base"
EMB_ID = "BAAI/bge-small-en-v1.5"
TOP_K = 5
SEED = 1337

# Reference points already measured, for the printout
REF_ORIGINAL, REF_GOLD, REF_GOLD_ANS = 0.546, 0.779, 0.936

PROMPT_BLIND = (
    "A multi-hop question needs several facts looked up in order.\n\n"
    "Question: {q}\n\n"
    "What is the LAST fact that must be looked up to answer this, once the "
    "earlier ones are known? Write it as a single short question. "
    "Reply with only that question, nothing else."
)

PROMPT_ANCHORED = (
    "A multi-hop question needs several facts looked up in order. Here is the "
    "question and a passage that answers part of it.\n\n"
    "Question: {q}\n\n"
    "Passage: {c1}\n\n"
    "Given what the passage tells you, what single follow-up question must be "
    "answered next to reach the final answer? Use specific names from the "
    "passage instead of descriptions like \"the person\". "
    "Reply with only that question, nothing else."
)

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


def seed_all():
    import random
    import numpy as np
    import torch
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def generate(prompt, model_id=DECOMPOSER, max_new_tokens=48):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    key = f"gen::{model_id}"
    if key not in _C:
        _env()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(dev).eval()
        _C[key] = (tok, mdl, dev)
        print(f"[load] {model_id}")
    tok, mdl, dev = _C[key]
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok([text], return_tensors="pt").to(dev)
    with torch.inference_mode():
        o = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         num_beams=1, pad_token_id=tok.eos_token_id)
    out = tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out.strip().split("\n")[0].strip().strip('"')


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


# ============================================================================
# Stage 1: let the LLM decompose
# ============================================================================

@app.function(gpu="L4", cpu=4, memory=32768, **COMMON)
def decompose_shard(lo: int, hi: int) -> str:
    _env(); seed_all()
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    out_path = R / "llmdecomp" / "subq" / f"{lo:05d}.jsonl"
    if out_path.exists():
        return str(out_path)

    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    keep = [r["qid"] for r in read_jsonl(R / "adv" / "splits" / "musique.jsonl")]
    ex_by_id = {str(e["id"]): e for e in ds}
    emb = SentenceTransformer(EMB_ID, device="cuda")

    rows = []
    for qid in keep[lo:hi]:
        ex = ex_by_id.get(qid)
        if ex is None:
            continue
        decomp = ex.get("question_decomposition") or []
        if len(decomp) < 2:
            continue
        q = str(ex["question"])
        paras = ex["paragraphs"]
        texts = [(str(p.get("title", "")) + ". " +
                  str(p.get("paragraph_text", ""))).strip() for p in paras]

        qv = emb.encode([q], show_progress_bar=False)[0]
        pv = emb.encode(texts, batch_size=32, show_progress_bar=False)
        cos = (pv @ qv) / (np.linalg.norm(pv, axis=1) * np.linalg.norm(qv) + 1e-9)
        anchor_i = int(np.argmax(cos))

        # the hop we evaluate on: the last annotated step with a support index
        target, prev = None, {}
        for i, st in enumerate(decomp, start=1):
            prev[i] = st.get("answer", "")
            if i > 1 and st.get("paragraph_support_idx") is not None:
                target = st
                break
        if target is None:
            continue

        sub_blind = generate(PROMPT_BLIND.format(q=q))
        sub_anch = generate(PROMPT_ANCHORED.format(
            q=q, c1=texts[anchor_i][:900]))

        rows.append({
            "qid": qid, "question": q, "n_hops": len(decomp),
            "anchor_idx": anchor_i,
            "anchor_is_gold": bool(paras[anchor_i].get("is_supporting", False)),
            "target_idx": int(target["paragraph_support_idx"]),
            "gold_subq": str(target.get("question", "")),
            "gold_subanswer": str(target.get("answer", "")),
            "sub_blind": sub_blind, "sub_anchored": sub_anch,
        })
    write_jsonl(out_path, rows)
    results.commit()
    return str(out_path)


# ============================================================================
# Stage 2: score every paragraph against each decomposition
# ============================================================================

@app.function(gpu="T4", cpu=4, memory=16384, **COMMON)
def score_decompositions() -> dict:
    _env(); seed_all()
    import pandas as pd
    from datasets import load_dataset
    import triver_core as tc

    files = sorted((R / "llmdecomp" / "subq").glob("*.jsonl"))
    sub = pd.DataFrame([r for f in files for r in read_jsonl(f)])
    print(f"{len(sub)} questions decomposed")

    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    ex_by_id = {str(e["id"]): e for e in ds}

    def prop_from_subq(s):
        """MuSiQue gold sub-questions are relational ('X >> relation'); LLM
        output is a natural question. Handle both."""
        s = str(s).strip()
        if ">>" in s:
            subj, rel = [x.strip() for x in s.split(">>", 1)]
            return f"The {rel} of {subj} is something."
        return tc.build_hypotheses(s, "prop_slot")[0]

    def resolve_placeholders(sub, ex):
        """Gold sub-questions carry #1, #2 for earlier hop answers. Leaving them
        literal handicaps the gold reference against LLM output, which has no
        placeholders, and it is not the quantity the decomposition experiment
        measured."""
        sub = str(sub)
        for i, st in enumerate(ex.get("question_decomposition") or [], start=1):
            sub = sub.replace("#%d" % i, str(st.get("answer", "")))
        return sub

    meta, p_blind, p_anch, p_gold = [], [], [], []
    for _, r in sub.iterrows():
        ex = ex_by_id.get(r.qid)
        if ex is None:
            continue
        h_b = prop_from_subq(r.sub_blind)
        h_a = prop_from_subq(r.sub_anchored)
        h_g = prop_from_subq(resolve_placeholders(r.gold_subq, ex))
        for j, p in enumerate(ex["paragraphs"]):
            t = (str(p.get("title", "")) + ". " +
                 str(p.get("paragraph_text", ""))).strip()
            p_blind.append((t, h_b)); p_anch.append((t, h_a)); p_gold.append((t, h_g))
            meta.append({"qid": r.qid, "n_hops": int(r.n_hops),
                         "anchor_is_gold": bool(r.anchor_is_gold),
                         "label": int(int(p.get("idx", j)) == int(r.target_idx))})

    print(f"scoring {len(p_blind)} pairs x 3 decompositions")
    s_b, s_a, s_g = nli(p_blind), nli(p_anch), nli(p_gold)
    rows = [{**m, "s_blind": a, "s_anchored": b, "s_gold": c}
            for m, a, b, c in zip(meta, s_b, s_a, s_g)]
    write_jsonl(R / "llmdecomp" / "scored_v2.jsonl", rows)
    results.commit()
    print(f"saved {len(rows)} rows")
    return {"n": len(rows)}


# ============================================================================
# Stage 3: AUC with intervals, and what fraction of the ceiling is reached
# ============================================================================

@app.function(cpu=16, memory=32768, **COMMON)
def analyze_llmdecomp() -> dict:
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata

    def fast_auc(y, s):
        y = np.asarray(y); s = np.asarray(s, float)
        n1 = int(y.sum()); n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            return float("nan")
        r = rankdata(s)
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def boot(y, s, q, n_boot=1000):
        y = np.asarray(y); s = np.asarray(s, float); q = np.asarray(q)
        pt = fast_auc(y, s)
        uq = np.unique(q); idx = {u: np.flatnonzero(q == u) for u in uq}
        rng = np.random.default_rng(SEED); b = []
        for _ in range(n_boot):
            pick = rng.choice(uq, len(uq), replace=True)
            ii = np.concatenate([idx[u] for u in pick])
            a = fast_auc(y[ii], s[ii])
            if np.isfinite(a):
                b.append(a)
        lo, hi = np.percentile(b, [2.5, 97.5])
        return pt, float(lo), float(hi), np.array(b)

    sc = R / "llmdecomp" / "scored_v2.jsonl"
    if not sc.exists():
        sc = R / "llmdecomp" / "scored.jsonl"
    print("reading " + sc.name)
    d = pd.DataFrame(read_jsonl(sc))
    out, boots = {}, {}
    for col, name in (("s_blind", "LLM, question only"),
                      ("s_anchored", "LLM, anchored on top-1 chunk"),
                      ("s_gold", "gold annotation")):
        v, lo, hi, bs = boot(d.label.values, d[col].values, d.qid.values)
        out[name] = {"auc": round(v, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
        boots[name] = bs
        print(f"  {name:34s} {v:.3f} [{lo:.3f}, {hi:.3f}]")

    best = max(out, key=lambda k: out[k]["auc"])
    span = REF_GOLD - REF_ORIGINAL
    for k in out:
        out[k]["fraction_of_ceiling"] = round(
            (out[k]["auc"] - REF_ORIGINAL) / span, 3)

    # paired bootstrap: does the deployable version beat the original question?
    rng = np.random.default_rng(SEED)
    uq = d.qid.unique(); idx = {u: np.flatnonzero(d.qid.values == u) for u in uq}
    diffs = []
    for _ in range(1000):
        pick = rng.choice(uq, len(uq), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        a = fast_auc(d.label.values[ii], d.s_anchored.values[ii])
        diffs.append(a)
    dlo, dhi = np.percentile(np.array(diffs) - REF_ORIGINAL, [2.5, 97.5])
    out["anchored_minus_original_question"] = {
        "delta": round(float(np.mean(diffs)) - REF_ORIGINAL, 4),
        "lo": round(float(dlo), 4), "hi": round(float(dhi), 4)}

    # per hop count
    for h, g in d.groupby("n_hops"):
        if g.qid.nunique() < 40:
            continue
        out[f"{int(h)}hop"] = {
            "anchored": round(fast_auc(g.label.values, g.s_anchored.values), 4),
            "gold": round(fast_auc(g.label.values, g.s_gold.values), 4),
            "n_q": int(g.qid.nunique())}

    # does a correct anchor matter?
    for flag, g in d.groupby("anchor_is_gold"):
        out[f"anchor_gold_{bool(flag)}"] = {
            "anchored_auc": round(fast_auc(g.label.values, g.s_anchored.values), 4),
            "n_q": int(g.qid.nunique())}

    AN = R / "results" / "analysis"; AN.mkdir(parents=True, exist_ok=True)
    (AN / "llm_decomposition.json").write_text(json.dumps(out, indent=2))

    TEX = R / "results" / "tex"; TEX.mkdir(parents=True, exist_ok=True)
    (TEX / "llm_decomp.tex").write_text("\n".join([
        "% generated by modal_llmdecomp.py",
        r"\begin{table}[t]", r"\centering",
        r"\caption{Identifying the paragraph supporting a later hop on MuSiQue, "
        r"by how the hypothesis is built. The LLM decomposer is "
        r"Qwen2.5-7B-Instruct with no fine-tuning and no gold annotation at "
        r"inference. Intervals are question-stratified bootstraps.}",
        r"\label{tab:llmdecomp}",
        r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
        r"\textbf{Hypothesis from} & \textbf{AUC \small{[95\% CI]}} & "
        r"\textbf{\% of ceiling}\\", r"\midrule",
        rf"the original question & {REF_ORIGINAL:.3f} & 0\%\\",
        *[rf"{k} & {v['auc']:.3f} \small{{[{v['lo']:.3f}, {v['hi']:.3f}]}} & "
          rf"{100*v['fraction_of_ceiling']:.0f}\%\\"
          for k, v in out.items() if isinstance(v, dict) and "fraction_of_ceiling" in v],
        r"\bottomrule", r"\end{tabular}", r"\end{table}"]))
    results.commit()

    print("\n" + json.dumps(out, indent=2))
    a = out["LLM, anchored on top-1 chunk"]
    print("\n" + "=" * 70)
    if a["auc"] >= 0.68:
        print(f"  DEPLOYABLE. An off-the-shelf decomposer reaches {a['auc']:.3f}, "
              f"{100*a['fraction_of_ceiling']:.0f}% of the way")
        print(f"  from the original question ({REF_ORIGINAL:.3f}) to the gold "
              f"annotation ({REF_GOLD:.3f}).")
        print("  The paper's positive result is now a method, not a ceiling.")
    elif a["auc"] >= 0.62:
        print(f"  PARTIAL. {a['auc']:.3f}, "
              f"{100*a['fraction_of_ceiling']:.0f}% of the ceiling. Report it and")
        print("  keep the ceiling framing, noting that decomposer quality is the")
        print("  bottleneck rather than the idea.")
    else:
        print(f"  WEAK. {a['auc']:.3f}. An off-the-shelf decomposer does not")
        print("  capture the effect. The gold result stays a ceiling, and")
        print("  building a decomposer good enough becomes the open problem.")
    print("=" * 70)

    with __import__("wandb").init(project="triver-rag", job_type="llm_decomp",
                                  name="llm_decomp", group="paper-v2", reinit=True):
        import wandb
        wandb.log({k: v["auc"] for k, v in out.items()
                   if isinstance(v, dict) and "auc" in v})
    return out


# ============================================================================

@app.function(cpu=4, memory=8192, **COMMON)
def show_examples(n: int = 6) -> None:
    """Print a few decompositions so the qualitative claim can be checked."""
    files = sorted((R / "llmdecomp" / "subq").glob("*.jsonl"))
    rows = [r for f in files for r in read_jsonl(f)][:n]
    for r in rows:
        print("-" * 78)
        print(f"Q          {r['question']}")
        print(f"gold subq  {r['gold_subq']}")
        print(f"LLM blind  {r['sub_blind']}")
        print(f"LLM anchor {r['sub_anchored']}")
        print(f"anchor was gold: {r['anchor_is_gold']}")


def _shards(n, size):
    return [(i, min(i + size, n)) for i in range(0, n, size)]


@app.local_entrypoint()
def main(stage: str = "decompose", n: int = 500, shard: int = 50):
    t0 = time.time()
    if stage in ("decompose", "all"):
        jobs = _shards(n, shard)
        print(f">> decomposing with {DECOMPOSER}: {len(jobs)} shards")
        list(decompose_shard.starmap(jobs))
        show_examples.remote()
    if stage in ("score", "all"):
        print(">> scoring every paragraph against each decomposition")
        print(score_decompositions.remote())
    if stage in ("score", "analyze", "all"):
        print(">> analysis")
        analyze_llmdecomp.remote()
    print(f"\nwall clock {(time.time()-t0)/60:.1f} min")
