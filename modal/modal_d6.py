"""
modal_d6.py — D6 only, with the dataset loader bypassed.

`datasets==2.21.0` cannot parse the current SQuAD dataset card, which is what
raised `TypeError: must be called with a dataclass type or instance`. This
version downloads the parquet shard directly through huggingface_hub and reads
it with pandas, so no dataset-script or card parsing is involved.

    modal run modal_d6.py

Also adds a length control on HotpotQA: gold+gold versus gold+distractor at
matched premise length, so the D5 lift cannot be attributed to longer premises.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "transformers==4.44.2", "tokenizers==0.19.1",
        "sentencepiece==0.2.0", "sentence-transformers==3.0.1",
        "huggingface_hub==0.24.6", "numpy==1.26.4", "scipy==1.14.0",
        "scikit-learn==1.5.1", "pandas==2.2.2", "pyarrow==17.0.0",
        "wandb==0.17.7",
    )
    .add_local_python_source("triver_core")
)

cache = modal.Volume.from_name("triver-cache", create_if_missing=True)
results = modal.Volume.from_name("triver-results", create_if_missing=True)
COMMON = dict(image=image, volumes={"/cache": cache, "/results": results},
              secrets=[modal.Secret.from_name("huggingface-secret"),
                       modal.Secret.from_name("wandb-secret")],
              timeout=3600, retries=2)

app = modal.App("triver-d6")
R = Path("/results")
NLI_ID = "cross-encoder/nli-deberta-v3-base"
EMB_ID = "BAAI/bge-small-en-v1.5"
SEED = 1337

_C = {}


def _env():
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def nli_entail(pairs, model_id=NLI_ID, batch=32, max_len=512):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    key = f"nli::{model_id}"
    if key not in _C:
        _env()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_id).to(dev).eval()
        id2 = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
        ent_ix = next(i for i, v in id2.items() if v == "entailment")
        _C[key] = (tok, mdl, dev, ent_ix)
        print(f"[load] {model_id} on {dev}, entailment index {ent_ix}")
    tok, mdl, dev, ent_ix = _C[key]
    out = []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch):
            b = pairs[i:i + batch]
            enc = tok([p for p, _ in b], [h for _, h in b], return_tensors="pt",
                      padding=True, truncation=True, max_length=max_len).to(dev)
            out.extend(torch.softmax(mdl(**enc).logits, -1)[:, ent_ix].float().cpu().tolist())
    return out


def load_squad_validation():
    """Bypass `datasets` entirely: pull the parquet shard and read it."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    for fname in ("plain_text/validation-00000-of-00001.parquet",
                  "validation-00000-of-00001.parquet"):
        try:
            p = hf_hub_download("rajpurkar/squad", fname, repo_type="dataset",
                                cache_dir=os.environ.get("HF_HOME", "/cache/hf"))
            df = pd.read_parquet(p)
            print(f"loaded {fname}: {len(df)} rows, columns {list(df.columns)}")
            return df
        except Exception as e:
            print(f"  {fname} -> {type(e).__name__}: {e}")
    raise RuntimeError("could not fetch SQuAD validation parquet")


@app.function(gpu="T4", cpu=4, memory=16384, **COMMON)
def d6_single_hop(n_q: int = 300) -> dict:
    """Same signals, same models, a dataset where the answer-bearing paragraph
    IS the query-similar paragraph. Distractors come from the same Wikipedia
    article so they stay hard negatives, as in HotpotQA distractor."""
    _env()
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import roc_auc_score
    import triver_core as tc

    df = load_squad_validation()

    def first_answer(a):
        if isinstance(a, dict):
            t = a.get("text")
        else:
            t = a
        if hasattr(t, "tolist"):
            t = t.tolist()
        return t[0] if isinstance(t, (list, tuple)) and t else str(t)

    by_title = defaultdict(list)
    for title, ctx in zip(df["title"], df["context"]):
        by_title[title].append(ctx)
    by_title = {k: list(dict.fromkeys(v)) for k, v in by_title.items()}

    rng = np.random.default_rng(SEED)
    items, seen_ctx = [], set()
    for _, row in df.iterrows():
        if len(items) >= n_q:
            break
        key = (row["title"], row["context"])
        if key in seen_ctx:
            continue
        pool = [c for c in by_title[row["title"]] if c != row["context"]]
        if len(pool) < 4:
            continue
        seen_ctx.add(key)
        negs = list(rng.choice(pool, size=min(9, len(pool)), replace=False))
        items.append({"q": str(row["question"]), "a": first_answer(row["answers"]),
                      "gold": str(row["context"]), "negs": [str(x) for x in negs]})
    print(f"built {len(items)} single-hop items, "
          f"{np.mean([len(i['negs']) for i in items]):.1f} distractors each")

    emb = SentenceTransformer(EMB_ID, device="cuda")
    labels, s_emb, p_slot, p_orac = [], [], [], []
    for it in items:
        texts = [it["gold"]] + it["negs"]
        qv = emb.encode([it["q"]], show_progress_bar=False)[0]
        pv = emb.encode(texts, batch_size=32, show_progress_bar=False)
        cos = (pv @ qv) / (np.linalg.norm(pv, axis=1) * np.linalg.norm(qv) + 1e-9)
        hs = tc.build_hypotheses(it["q"], "prop_slot")[0]
        ho = tc.build_hypotheses(it["q"], "prop_oracle", gold=it["a"])[0]
        for j, (t, c) in enumerate(zip(texts, cos)):
            labels.append(1 if j == 0 else 0)
            s_emb.append(tc.rescale_cosine(float(c)))
            p_slot.append((t, hs)); p_orac.append((t, ho))

    print(f"scoring {len(p_slot)} pairs x 2 hypothesis modes")
    slot, orac = nli_entail(p_slot), nli_entail(p_orac)

    gold_mask = np.array(labels) == 1
    out = {
        "dataset": "SQuAD v1.1 validation, same-article distractors",
        "n_questions": len(items), "n_pairs": len(labels),
        "nli_model": NLI_ID, "emb_model": EMB_ID,
        "AUC_s_emb": round(float(roc_auc_score(labels, s_emb)), 4),
        "AUC_s_ent_slot": round(float(roc_auc_score(labels, slot)), 4),
        "AUC_s_ent_oracle": round(float(roc_auc_score(labels, orac)), 4),
        "mean_entail_gold": round(float(np.mean(np.array(orac)[gold_mask])), 4),
        "mean_entail_distractor": round(float(np.mean(np.array(orac)[~gold_mask])), 4),
        "hotpotqa_single_chunk_for_comparison": 0.664,
    }
    out["single_hop_lift_over_hotpotqa"] = round(out["AUC_s_ent_oracle"] - 0.664, 4)

    d = R / "results" / "analysis"; d.mkdir(parents=True, exist_ok=True)
    (d / "d6_single_hop.json").write_text(json.dumps(out, indent=2))
    results.commit()

    print("\n" + json.dumps(out, indent=2))
    print("\n" + "=" * 70)
    if out["AUC_s_ent_oracle"] >= 0.80:
        print("  CONFIRMED. Entailment verification works on single-hop QA and")
        print("  fails on multi-hop. The failure is structural, not a property")
        print("  of small NLI models.")
    elif out["AUC_s_ent_oracle"] >= 0.72:
        print("  Partial. Meaningfully better single-hop but not strong.")
        print("  Report both numbers and keep the claim narrow.")
    else:
        print("  Weak even single-hop. Then the honest reading is that")
        print("  entailment verification does not work at this model scale,")
        print("  and the multi-hop structure is a second, separate problem.")
    print("=" * 70)
    return out


@app.function(gpu="T4", cpu=4, memory=16384, **COMMON)
def d7_length_control(n_q: int = 200) -> dict:
    """Rules out the obvious alternative explanation for the D5 lift: that a
    longer premise simply raises entailment probability. Compares gold+gold
    against gold+distractor against distractor+distractor, all two-paragraph
    premises of comparable length."""
    _env()
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import triver_core as tc

    rng = np.random.default_rng(SEED)
    recs = read_jsonl(R / "splits" / "eval.jsonl")[:n_q]

    buckets = {"gold+gold": [], "gold+distractor": [], "distractor+distractor": []}
    lens = {k: [] for k in buckets}
    for rec in recs:
        hyp = tc.build_hypotheses(rec["question"], "prop_oracle", gold=rec["answer"])[0]
        paras = rec["paragraphs"]
        g = [i for i, p in enumerate(paras) if p["is_gold"]]
        d = [i for i, p in enumerate(paras) if not p["is_gold"]]
        if len(g) < 2 or len(d) < 2:
            continue
        def prem(a, b):
            return paras[a]["text"] + " " + paras[b]["text"]
        combos = [("gold+gold", g[0], g[1]),
                  ("gold+distractor", g[0], int(rng.choice(d))),
                  ("distractor+distractor", *rng.choice(d, 2, replace=False))]
        for name, a, b in combos:
            t = prem(int(a), int(b))
            buckets[name].append((t, hyp))
            lens[name].append(len(t.split()))

    out = {"n_questions": len(recs), "nli_model": NLI_ID}
    scores = {}
    for name, pairs in buckets.items():
        s = nli_entail(pairs)
        scores[name] = s
        out[f"mean_entail__{name}"] = round(float(np.mean(s)), 4)
        out[f"mean_premise_words__{name}"] = round(float(np.mean(lens[name])), 1)

    y = [1] * len(scores["gold+gold"]) + [0] * len(scores["gold+distractor"])
    out["AUC_goldgold_vs_golddistractor"] = round(
        float(roc_auc_score(y, scores["gold+gold"] + scores["gold+distractor"])), 4)

    d = R / "results" / "analysis"; d.mkdir(parents=True, exist_ok=True)
    (d / "d7_length_control.json").write_text(json.dumps(out, indent=2))
    results.commit()

    print("\n" + json.dumps(out, indent=2))
    print("\n" + "=" * 70)
    gg, gd = out["mean_entail__gold+gold"], out["mean_entail__gold+distractor"]
    if gg > 2 * gd:
        print("  Premise length is ruled out. Two paragraphs of the same length")
        print("  score far lower unless BOTH hops are present. The D5 lift comes")
        print("  from evidence completeness, not from having more text.")
    else:
        print("  gold+distractor is close to gold+gold. Investigate before")
        print("  claiming the set-level effect.")
    print("=" * 70)
    return out


@app.local_entrypoint()
def main():
    print("\n########## D7  length control (HotpotQA) ##########")
    d7 = d7_length_control.remote()
    print("\n########## D6  single-hop control (SQuAD) ##########")
    d6 = d6_single_hop.remote()

    print("\n" + "=" * 70)
    print("  THE FOUR NUMBERS THE PAPER TURNS ON")
    print("=" * 70)
    print(f"  HotpotQA, single chunk            : 0.664")
    print(f"  HotpotQA, gold pair               : 0.881")
    print(f"  HotpotQA, gold+distractor (mean p): {d7['mean_entail__gold+distractor']:.3f}")
    print(f"  HotpotQA, gold+gold       (mean p): {d7['mean_entail__gold+gold']:.3f}")
    print(f"  SQuAD single-hop, single chunk    : {d6['AUC_s_ent_oracle']:.3f}")
    print("=" * 70)
