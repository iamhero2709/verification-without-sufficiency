"""
modal_advanced.py — take the paper from a solid negative result to a strong one.

Five additions, each closing a hole a reviewer will otherwise open.

  A  Generalisation.  Repeat the whole analysis on 2WikiMultihopQA and MuSiQue.
     Turns "this happens on HotpotQA" into "this is a property of multi-hop
     retrieval." Highest reviewer value.

  B  Hop scaling.  MuSiQue labels each question 2hop / 3hop / 4hop. If the
     mechanism is right, per-chunk AUC should fall monotonically with hop count.
     One figure that proves the whole argument.

  C  Conditional verifier.  The paper says "condition on the reasoning state,
     not the question" and never tests it. Anchor on the top embedding chunk,
     which we measured at 0.927 AUC on hop-1 evidence, then score the rest
     against premise = c1 + cj. Four NLI calls instead of ten. Highest novelty.

  D  Threshold sweep.  The per-chunk gate used tau = 0.35. Sweep it and show no
     threshold rescues it. Closes the obvious objection. Mostly free.

  E  Retriever generalisation.  Three embedding models, same split. Shows the
     directional bias is not an artefact of bge-small.

    modal run modal_advanced.py --stage data       # ~4 min,  $0.30
    modal run modal_advanced.py --stage signals    # ~25 min, $1.50
    modal run modal_advanced.py --stage auc        # free      <- GATE
    modal run modal_advanced.py --stage cond       # ~8 min,  $1.00  <- GATE
    modal run modal_advanced.py --stage gen        # ~60 min, $5.00
    modal run modal_advanced.py --stage extras     # ~10 min, $0.60  (D + E)
    modal run modal_advanced.py --stage analyze    # free

Budget: about $8.50 plus buffer. Two gates stop you early if a premise fails.
"""
from __future__ import annotations

import itertools
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
        "accelerate==0.33.0", "datasets==2.21.0", "huggingface_hub==0.24.6",
        "numpy==1.26.4", "scipy==1.14.0", "scikit-learn==1.5.1",
        "pandas==2.2.2", "pyarrow==17.0.0", "matplotlib==3.9.2", "wandb==0.17.7",
    )
    .add_local_python_source("triver_core")
)

cache = modal.Volume.from_name("triver-cache", create_if_missing=True)
results = modal.Volume.from_name("triver-results", create_if_missing=True)
COMMON = dict(image=image, volumes={"/cache": cache, "/results": results},
              secrets=[modal.Secret.from_name("huggingface-secret"),
                       modal.Secret.from_name("wandb-secret")],
              timeout=60 * 60 * 2, retries=2)

app = modal.App("triver-advanced")
R = Path("/results")

NLI_ID = "cross-encoder/nli-deberta-v3-base"
EMB_ID = "BAAI/bge-small-en-v1.5"
EMB_ALTS = ["BAAI/bge-small-en-v1.5", "intfloat/e5-base-v2", "thenlper/gte-base"]
RERANK_ID = "BAAI/bge-reranker-base"
GEN_IDS = ["Qwen/Qwen2.5-1.5B-Instruct"]
TOP_K = 5
N_PER_DS = 500
SEED = 1337
PC_TAUS = [0.10, 0.20, 0.30, 0.35, 0.45, 0.60, 0.75]

DATASETS = ["hotpotqa", "2wiki", "musique"]
CONFIGS = ["B1", "B2", "B4", "PC", "SET", "COND", "B5"]

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


# ============================================================================
# Robust dataset loading
# ============================================================================

def _try_load(repo, config=None, split="validation"):
    from datasets import load_dataset
    return load_dataset(repo, config, split=split) if config else \
        load_dataset(repo, split=split)


def _parquet_fallback(repo, split_hint=("valid", "dev")):
    """List repo files, find a parquet matching the split, read with pandas.
    Used when the `datasets` loader cannot parse the dataset card."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    cands = [f for f in files if f.endswith(".parquet")
             and any(h in f.lower() for h in split_hint)]
    if not cands:
        cands = [f for f in files if f.endswith((".json", ".jsonl"))
                 and any(h in f.lower() for h in split_hint)]
    if not cands:
        raise RuntimeError(f"{repo}: no split file found among {files[:20]}")
    path = hf_hub_download(repo, sorted(cands)[0], repo_type="dataset",
                           cache_dir=os.environ.get("HF_HOME", "/cache/hf"))
    print(f"  fallback: {repo} -> {sorted(cands)[0]}")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_json(path, lines=path.endswith(".jsonl"))


def load_raw(name):
    """Return an iterable of raw records for one dataset."""
    _env()
    tries = {
        "hotpotqa": [("hotpotqa/hotpot_qa", "distractor")],
        "2wiki":    [("xanhho/2WikiMultihopQA", None),
                     ("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35", None)],
        "musique":  [("dgslibisey/MuSiQue", None), ("bdsaglam/musique", None)],
    }[name]
    for repo, cfg in tries:
        try:
            ds = _try_load(repo, cfg)
            print(f"  loaded {repo} via datasets: {len(ds)} rows")
            return list(ds), repo
        except Exception as e:
            print(f"  {repo} datasets loader -> {type(e).__name__}: {str(e)[:120]}")
            try:
                df = _parquet_fallback(repo)
                print(f"  loaded {repo} via parquet: {len(df)} rows")
                return df.to_dict("records"), repo
            except Exception as e2:
                print(f"  {repo} parquet fallback -> {type(e2).__name__}: {str(e2)[:120]}")
    raise RuntimeError(f"could not load {name}")


def _decode(x, depth=0):
    """Coerce whatever parquet/JSON handed us into plain Python structures.
    Parquet round-trips can turn a list of lists into a JSON string, a numpy
    array, or a dict of arrays, and each dataset does something different."""
    import numpy as np
    if depth > 6:
        return x
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", "replace")
    if isinstance(x, str):
        t = x.strip()
        if t[:1] in ("[", "{"):
            try:
                return _decode(json.loads(t), depth + 1)
            except Exception:
                return x
        return x
    if isinstance(x, np.ndarray):
        return [_decode(v, depth + 1) for v in x.tolist()]
    if isinstance(x, dict):
        return {k: _decode(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_decode(v, depth + 1) for v in x]
    if hasattr(x, "tolist"):
        return _decode(x.tolist(), depth + 1)
    return x


def _as_text(s):
    """Sentences may be a list, a nested list, or an already-joined string."""
    if isinstance(s, str):
        return s
    if isinstance(s, (list, tuple)):
        return " ".join(_as_text(v) for v in s)
    return str(s)


def _title_sent_pairs(ctx):
    """Return [(title, sentences)] from any of the shapes 2Wiki/HotpotQA use."""
    ctx = _decode(ctx)
    if isinstance(ctx, dict):
        titles = ctx.get("title")
        sents = ctx.get("content", ctx.get("sentences", ctx.get("paragraphs")))
        if titles is None or sents is None:
            return []
        return list(zip(titles, sents))
    if isinstance(ctx, (list, tuple)) and ctx:
        first = ctx[0]
        if isinstance(first, dict):
            return [(c.get("title", ""),
                     c.get("content", c.get("sentences", c.get("paragraph_text", ""))))
                    for c in ctx]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return [(c[0], c[1]) for c in ctx if len(c) >= 2]
    return []


def _gold_titles(sf):
    sf = _decode(sf)
    if isinstance(sf, dict):
        return {str(t) for t in sf.get("title", [])}
    if isinstance(sf, (list, tuple)):
        out = set()
        for e in sf:
            if isinstance(e, dict):
                out.add(str(e.get("title", "")))
            elif isinstance(e, (list, tuple)) and e:
                out.add(str(e[0]))
            elif isinstance(e, str):
                out.add(e)
        return out
    return set()


@app.function(cpu=4, memory=16384, **COMMON)
def inspect_datasets(n: int = 2) -> dict:
    """Print the actual schema before writing another normaliser by guesswork."""
    _env()
    out = {}
    for name in ("2wiki", "musique"):
        try:
            raw, repo = load_raw(name)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"\n### {name}: FAILED  {e}")
            continue
        ex = raw[0]
        info = {"repo": repo, "n_rows": len(raw), "fields": {}}
        print(f"\n### {name}  ({repo}, {len(raw)} rows)")
        for k, v in (ex.items() if isinstance(ex, dict) else []):
            d = _decode(v)
            kind = type(v).__name__
            if isinstance(d, (list, tuple)):
                shape = f"list[{len(d)}] of {type(d[0]).__name__ if d else '?'}"
                sample = str(d[0])[:160] if d else ""
            elif isinstance(d, dict):
                shape = f"dict keys={list(d)[:6]}"
                sample = str({kk: str(vv)[:60] for kk, vv in list(d.items())[:2]})[:200]
            else:
                shape = kind
                sample = str(d)[:160]
            info["fields"][k] = {"raw_type": kind, "decoded": shape, "sample": sample}
            print(f"  {k:24s} {kind:12s} -> {shape}")
            print(f"    {sample}")
        pairs = _title_sent_pairs(ex.get("context")) if isinstance(ex, dict) else []
        info["title_sent_pairs_found"] = len(pairs)
        if pairs:
            print(f"  _title_sent_pairs -> {len(pairs)} pairs, first title: {str(pairs[0][0])[:60]}")
        gt = _gold_titles(ex.get("supporting_facts")) if isinstance(ex, dict) else set()
        info["gold_titles_found"] = len(gt)
        if gt:
            print(f"  _gold_titles -> {list(gt)[:3]}")
        out[name] = info
    (R / "adv").mkdir(parents=True, exist_ok=True)
    (R / "adv" / "schema.json").write_text(json.dumps(out, indent=2, default=str))
    results.commit()
    return out


def normalise(name, ex, i):
    """Map one raw record into the common schema. Returns None if unusable."""
    def mk(qid, question, answer, paras, qtype, n_hops):
        gold = [p for p in paras if p["is_gold"]]
        if len(gold) < 2 or len(paras) < 4:
            return None
        return {"qid": str(qid), "question": str(question), "answer": str(answer),
                "qtype": qtype, "n_hops": n_hops, "paragraphs": paras}

    if name == "hotpotqa":
        titles, sents = ex["context"]["title"], ex["context"]["sentences"]
        gold_t = set(ex["supporting_facts"]["title"])
        paras = [{"pid": f'{ex["id"]}::{j}', "title": t,
                  "text": (t + ". " + " ".join(s)).strip(),
                  "is_gold": bool(t in gold_t)}
                 for j, (t, s) in enumerate(zip(titles, sents))]
        return mk(ex["id"], ex["question"], ex["answer"], paras, ex.get("type", "?"), 2)

    if name == "2wiki":
        pairs = _title_sent_pairs(ex.get("context"))
        gold_t = _gold_titles(ex.get("supporting_facts"))
        if not pairs or not gold_t:
            return None
        qid = str(ex.get("_id", ex.get("id", f"2wiki_{i}")))
        paras = [{"pid": f"{qid}::{j}", "title": str(t),
                  "text": (str(t) + ". " + _as_text(s)).strip(),
                  "is_gold": bool(str(t) in gold_t)}
                 for j, (t, s) in enumerate(pairs)]
        return mk(qid, ex.get("question"), ex.get("answer"), paras,
                  str(ex.get("type", "?")), 2)

    if name == "musique":
        if ex.get("answerable") is False:
            return None                      # MuSiQue-Full has unanswerable items
        ps = _decode(ex.get("paragraphs"))
        if not isinstance(ps, (list, tuple)) or not ps or not isinstance(ps[0], dict):
            return None
        qid = str(ex.get("id", f"musique_{i}"))
        m = re.match(r"(\d)hop", qid)
        n_hops = int(m.group(1)) if m else 2

        # `is_supporting` is the primary label; question_decomposition carries
        # paragraph_support_idx as a fallback if the field is absent.
        support_idx = set()
        for d in _decode(ex.get("question_decomposition")) or []:
            if isinstance(d, dict) and d.get("paragraph_support_idx") is not None:
                support_idx.add(int(d["paragraph_support_idx"]))

        paras = []
        for j, p in enumerate(ps):
            gold = bool(p.get("is_supporting", False))
            if not gold and support_idx:
                gold = int(p.get("idx", j)) in support_idx
            paras.append({"pid": f"{qid}::{j}",
                          "title": str(p.get("title", "")),
                          "text": (str(p.get("title", "")) + ". " +
                                   str(p.get("paragraph_text", ""))).strip(),
                          "is_gold": gold})
        return mk(qid, ex["question"], ex["answer"], paras, f"{n_hops}hop", n_hops)

    raise ValueError(name)


@app.function(cpu=8, memory=32768, **{**COMMON, "retries": 0})
def build_splits() -> dict:
    _env(); seed_all()
    import numpy as np

    out = {}
    for name in DATASETS:
        if name == "hotpotqa":
            src = R / "splits" / "eval.jsonl"
            if src.exists():
                recs = read_jsonl(src)
                for r in recs:
                    r.setdefault("qtype", "?"); r.setdefault("n_hops", 2)
                write_jsonl(R / "adv" / "splits" / "hotpotqa.jsonl", recs)
                out["hotpotqa"] = {"n": len(recs), "source": "reused eval split"}
                print(f"hotpotqa: reusing {len(recs)} eval questions")
                continue
        raw, repo = load_raw(name)
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(raw))
        recs, skipped, errors = [], 0, {}
        for i in idx:
            try:
                r = normalise(name, raw[int(i)], int(i))
            except Exception as e:
                key = f"{type(e).__name__}: {str(e)[:60]}"
                errors[key] = errors.get(key, 0) + 1
                r = None
            if r:
                recs.append(r)
            else:
                skipped += 1
            if len(recs) >= N_PER_DS:
                break
        if not recs:
            print(f"  {name}: NO usable records. errors={errors}")
            out[name] = {"n": 0, "repo": repo, "skipped": skipped, "errors": errors}
            continue
        write_jsonl(R / "adv" / "splits" / f"{name}.jsonl", recs)
        hops = {}
        for r in recs:
            hops[r["n_hops"]] = hops.get(r["n_hops"], 0) + 1
        out[name] = {"n": len(recs), "repo": repo, "skipped": skipped, "errors": errors,
                     "mean_paragraphs": float(np.mean([len(r["paragraphs"]) for r in recs])),
                     "mean_gold": float(np.mean([sum(p["is_gold"] for p in r["paragraphs"])
                                                 for r in recs])),
                     "hop_counts": hops}
        print(f"{name}: {out[name]}")
    (R / "adv").mkdir(parents=True, exist_ok=True)
    (R / "adv" / "splits_meta.json").write_text(json.dumps(out, indent=2))
    results.commit()
    return out


# ============================================================================
# Models
# ============================================================================

def get_emb(model_id=EMB_ID):
    key = f"emb::{model_id}"
    if key not in _C:
        _env()
        import torch
        from sentence_transformers import SentenceTransformer
        _C[key] = SentenceTransformer(
            model_id, device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"[load] {model_id}")
    return _C[key]


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


def rerank(query, texts):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if "rr" not in _C:
        _env()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(RERANK_ID)
        mdl = AutoModelForSequenceClassification.from_pretrained(RERANK_ID).to(dev).eval()
        _C["rr"] = (tok, mdl, dev)
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
        print(f"[load] {model_id}")
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
# A + B  signals on every dataset
# ============================================================================

@app.function(gpu="T4", cpu=4, memory=16384, **COMMON)
def signals_shard(ds: str, lo: int, hi: int) -> str:
    _env(); seed_all()
    import numpy as np
    import triver_core as tc

    out_path = R / "adv" / "signals" / ds / f"{lo:05d}.jsonl"
    if out_path.exists():
        return str(out_path)

    recs = read_jsonl(R / "adv" / "splits" / f"{ds}.jsonl")[lo:hi]
    emb = get_emb()
    rows = []
    for rec in recs:
        q, gold = rec["question"], rec["answer"]
        paras = rec["paragraphs"]
        texts = [p["text"] for p in paras]
        qv = emb.encode([q], show_progress_bar=False)[0]
        pv = emb.encode(texts, batch_size=32, show_progress_bar=False)
        cos = (pv @ qv) / (np.linalg.norm(pv, axis=1) * np.linalg.norm(qv) + 1e-9)
        h_slot = tc.build_hypotheses(q, "prop_slot")[0]
        h_orac = tc.build_hypotheses(q, "prop_oracle", gold=gold)[0]
        e_slot = nli([(t, h_slot) for t in texts])
        e_orac = nli([(t, h_orac) for t in texts])
        na = tc.normalize_answer(gold)
        order = np.argsort(-cos)
        rank = {int(j): int(r) for r, j in enumerate(order)}
        for j, p in enumerate(paras):
            rows.append({
                "dataset": ds, "qid": rec["qid"], "pid": p["pid"],
                "qtype": rec.get("qtype", "?"), "n_hops": rec.get("n_hops", 2),
                "is_gold": p["is_gold"], "retriever_rank": rank[j],
                "retriever_score": float(cos[j]),
                "s_emb": tc.rescale_cosine(float(cos[j])),
                "s_ent_slot": e_slot[j], "s_ent_oracle": e_orac[j],
                "answer_bearing": bool(na not in ("yes", "no", "")
                                       and na in tc.normalize_answer(p["text"])),
            })
    write_jsonl(out_path, rows)
    results.commit()
    return str(out_path)


@app.function(cpu=8, memory=32768, **COMMON)
def auc_tables() -> dict:
    """GATE 1. Does the HotpotQA pattern reproduce on other multi-hop data,
    and does it worsen with hop count?"""
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    frames = []
    for ds in DATASETS:
        d = R / "adv" / "signals" / ds
        if d.exists():
            frames.append(pd.DataFrame([r for f in sorted(d.glob("*.jsonl"))
                                        for r in read_jsonl(f)]))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(R / "adv" / "signals.parquet")

    def split_auc(d, col):
        dis = d[~d.is_gold]
        def a(pos):
            if len(pos) < 20 or len(dis) < 20:
                return float("nan")
            return float(roc_auc_score([1] * len(pos) + [0] * len(dis),
                                       list(pos[col]) + list(dis[col])))
        return (float(roc_auc_score(d.is_gold, d[col])),
                a(d[d.is_gold & d.answer_bearing]),
                a(d[d.is_gold & ~d.answer_bearing]))

    rows = []
    for ds, d in df.groupby("dataset"):
        for col in ("s_emb", "s_ent_slot", "s_ent_oracle"):
            all_, bear, brid = split_auc(d, col)
            rows.append({"dataset": ds, "n_questions": int(d.qid.nunique()),
                         "signal": col, "AUC_all": all_,
                         "AUC_answer_bearing": bear, "AUC_bridge_only": brid,
                         "deficit": brid - bear})
    # per question type, which 2Wiki labels four ways and HotpotQA two
    type_rows = []
    for (ds, qt), d in df.groupby(["dataset", "qtype"]):
        if d.qid.nunique() < 30:
            continue
        for col in ("s_emb", "s_ent_slot"):
            all_, bear, brid = split_auc(d, col)
            type_rows.append({"dataset": ds, "qtype": str(qt),
                              "n_questions": int(d.qid.nunique()), "signal": col,
                              "AUC_all": all_, "AUC_answer_bearing": bear,
                              "AUC_bridge_only": brid, "deficit": brid - bear})
    if type_rows:
        pd.DataFrame(type_rows).to_csv(
            R / "results" / "analysis" / "adv_auc_by_qtype.csv", index=False)
        print("\\n--- AUC by question type " + "-" * 44)
        print(pd.DataFrame(type_rows).to_string(
            index=False, float_format=lambda x: f"{x:.3f}"))
    t = pd.DataFrame(rows)
    t.to_csv(R / "results" / "analysis" / "adv_auc_by_dataset.csv", index=False)

    hop_rows = []
    mq = df[df.dataset == "musique"]
    for h, d in mq.groupby("n_hops"):
        for col in ("s_emb", "s_ent_slot", "s_ent_oracle"):
            all_, bear, brid = split_auc(d, col)
            hop_rows.append({"n_hops": int(h), "n_questions": int(d.qid.nunique()),
                             "signal": col, "AUC_all": all_,
                             "AUC_answer_bearing": bear, "AUC_bridge_only": brid})
    ht = pd.DataFrame(hop_rows)
    ht.to_csv(R / "results" / "analysis" / "adv_auc_by_hops.csv", index=False)
    results.commit()

    print("\n--- AUC by dataset " + "-" * 50)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if len(ht):
        print("\n--- AUC by hop count (MuSiQue) " + "-" * 38)
        print(ht.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    ent = t[t.signal == "s_ent_slot"].set_index("dataset").AUC_all.to_dict()
    print("\n" + "=" * 70)
    if all(v < 0.75 for v in ent.values()) and len(ent) >= 2:
        print("  REPRODUCED. Per-chunk entailment is weak on every multi-hop")
        print("  dataset tested. The claim generalises beyond HotpotQA.")
    else:
        print(f"  MIXED: {ent}. Report per dataset and narrow the claim.")
    if len(ht) >= 2:
        e = ht[ht.signal == "s_ent_slot"].sort_values("n_hops")
        mono = all(x >= y for x, y in zip(e.AUC_all, e.AUC_all[1:]))
        print(f"  hop scaling: {dict(zip(e.n_hops, e.AUC_all.round(3)))}"
              f"  {'monotone decreasing' if mono else 'not monotone'}")
    print("=" * 70)

    with __import__("wandb").init(project="triver-rag", job_type="adv_auc",
                                  name="adv_auc", group="paper-v2", reinit=True):
        import wandb
        wandb.log({"auc_by_dataset": wandb.Table(dataframe=t)})
        if len(ht):
            wandb.log({"auc_by_hops": wandb.Table(dataframe=ht)})
    return {"by_dataset": t.to_dict("records"), "by_hops": ht.to_dict("records")}


# ============================================================================
# C  conditional verifier
# ============================================================================

def cond_select(rec, sig, k=TOP_K):
    """Anchor on the top embedding chunk, then pick the partner that best
    completes the evidence. Four NLI calls instead of ten."""
    import triver_core as tc
    top = sorted(rec["paragraphs"], key=lambda p: -sig[p["pid"]]["retriever_score"])[:k]
    if len(top) < 2:
        return top, None
    c1, rest = top[0], top[1:]
    h = tc.build_hypotheses(rec["question"], "prop_slot")[0]
    sc = nli([(c1["text"] + " " + p["text"], h) for p in rest])
    c2 = rest[int(max(range(len(sc)), key=lambda i: sc[i]))]
    return [c1, c2], max(sc)


@app.function(gpu="T4", cpu=4, memory=16384, **COMMON)
def cond_shard(ds: str, lo: int, hi: int) -> str:
    _env(); seed_all()
    import itertools as it
    import pandas as pd
    import triver_core as tc

    out_path = R / "adv" / "cond" / ds / f"{lo:05d}.jsonl"
    if out_path.exists():
        return str(out_path)

    sig = pd.read_parquet(R / "adv" / "signals.parquet")
    sig = sig[sig.dataset == ds].set_index("pid").to_dict("index")
    recs = read_jsonl(R / "adv" / "splits" / f"{ds}.jsonl")[lo:hi]

    rows = []
    for rec in recs:
        paras = rec["paragraphs"]
        top = sorted(paras, key=lambda p: -sig[p["pid"]]["retriever_score"])[:TOP_K]
        n_gold_avail = sum(p["is_gold"] for p in paras)
        h = tc.build_hypotheses(rec["question"], "prop_slot")[0]

        # conditional: anchor + best partner
        sel_cond, _ = cond_select(rec, sig)
        # unconditional set: argmax over all pairs
        combos = list(it.combinations(range(len(top)), 2))
        sc = nli([(top[a]["text"] + " " + top[b]["text"], h) for a, b in combos])
        a, b = combos[int(max(range(len(sc)), key=lambda i: sc[i]))]
        sel_set = [top[a], top[b]]

        rows.append({
            "dataset": ds, "qid": rec["qid"], "n_hops": rec.get("n_hops", 2),
            "anchor_is_gold": bool(top[0]["is_gold"]),
            "cond_gold_kept": sum(p["is_gold"] for p in sel_cond),
            "set_gold_kept": sum(p["is_gold"] for p in sel_set),
            "gold_available": n_gold_avail,
            "gold_in_top_k": sum(p["is_gold"] for p in top),
            "cond_pids": [p["pid"] for p in sel_cond],
            "set_pids": [p["pid"] for p in sel_set],
            "nli_calls_cond": len(top) - 1, "nli_calls_set": len(combos),
        })
    write_jsonl(out_path, rows)
    results.commit()
    return str(out_path)


@app.function(cpu=8, memory=16384, **COMMON)
def cond_gate() -> dict:
    """GATE 2. Does anchoring beat blind pair search at picking evidence?"""
    import numpy as np
    import pandas as pd

    frames = []
    for ds in DATASETS:
        d = R / "adv" / "cond" / ds
        if d.exists():
            frames.append(pd.DataFrame([r for f in sorted(d.glob("*.jsonl"))
                                        for r in read_jsonl(f)]))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(R / "adv" / "cond.parquet")

    rows = []
    for ds, d in df.groupby("dataset"):
        ceil = float((d.gold_in_top_k >= 2).mean())
        rows.append({
            "dataset": ds, "n": len(d),
            "anchor_is_gold": float(d.anchor_is_gold.mean()),
            "both_gold_reachable": ceil,
            "COND_recall": float((d.cond_gold_kept / d.gold_available.clip(lower=1)).mean()),
            "SET_recall": float((d.set_gold_kept / d.gold_available.clip(lower=1)).mean()),
            "COND_both_gold": float((d.cond_gold_kept >= 2).mean()),
            "SET_both_gold": float((d.set_gold_kept >= 2).mean()),
            "nli_calls_cond": float(d.nli_calls_cond.mean()),
            "nli_calls_set": float(d.nli_calls_set.mean()),
        })
    t = pd.DataFrame(rows)
    t["recall_lift"] = t.COND_recall - t.SET_recall
    t.to_csv(R / "results" / "analysis" / "adv_cond_gate.csv", index=False)
    results.commit()

    print("\n" + t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    hp = t[t.dataset == "hotpotqa"]
    print("\n" + "=" * 70)
    if len(hp) and float(hp.recall_lift.iloc[0]) > 0.05:
        print(f"  CONDITIONING WINS. Gold recall {float(hp.COND_recall.iloc[0]):.3f} against")
        print(f"  {float(hp.SET_recall.iloc[0]):.3f} for blind pair search, at "
              f"{float(hp.nli_calls_cond.iloc[0]):.0f} NLI calls instead of "
              f"{float(hp.nli_calls_set.iloc[0]):.0f}.")
        print("  Run --stage gen.")
    else:
        print("  No recall gain from anchoring. Report it as a negative and")
        print("  skip the end-to-end run for COND.")
    print("=" * 70)

    with __import__("wandb").init(project="triver-rag", job_type="adv_cond",
                                  name="adv_cond", group="paper-v2", reinit=True):
        import wandb; wandb.log({"cond_gate": wandb.Table(dataframe=t)})
    return {"table": t.to_dict("records")}


# ============================================================================
# generation
# ============================================================================

def select(cfg, rec, sig, pairs_by_q, tau=0.35):
    import triver_core as tc
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
    if cfg.startswith("PC"):
        t = float(cfg.split("@")[1]) if "@" in cfg else tau
        keep = [p for p in top if sig[p["pid"]]["s_ent_slot"] >= t]
        return keep if keep else top[:1]
    if cfg == "SET":
        r = pairs_by_q[rec["qid"]]
        return [by_pid[r["set_pids"][0]], by_pid[r["set_pids"][1]]]
    if cfg == "COND":
        r = pairs_by_q[rec["qid"]]
        return [by_pid[r["cond_pids"][0]], by_pid[r["cond_pids"][1]]]
    raise ValueError(cfg)


@app.function(gpu="L4", cpu=4, memory=32768, **COMMON)
def gen_shard(ds: str, cfg: str, lo: int, hi: int, model_id: str) -> str:
    _env(); seed_all()
    import pandas as pd
    import triver_core as tc

    tag = model_id.split("/")[-1]
    out_path = R / "results" / "adv_gen" / ds / tag / cfg.replace("@", "_") / f"{lo:05d}.jsonl"
    if out_path.exists():
        return str(out_path)

    sig = pd.read_parquet(R / "adv" / "signals.parquet")
    sig = sig[sig.dataset == ds].set_index("pid").to_dict("index")
    cdf = pd.read_parquet(R / "adv" / "cond.parquet")
    pairs_by_q = {r["qid"]: r for r in cdf[cdf.dataset == ds].to_dict("records")}
    recs = read_jsonl(R / "adv" / "splits" / f"{ds}.jsonl")[lo:hi]
    # SET and COND select two chunks, so a 3- or 4-hop answer is unreachable for
    # them by construction. Restricting generation to 2-hop questions keeps the
    # comparison fair; the AUC analysis uses every hop count.
    recs = [r for r in recs if int(r.get("n_hops", 2)) == 2]
    if not recs:
        write_jsonl(out_path, [])
        results.commit()
        return str(out_path)

    rows = []
    for rec in recs:
        kept = select(cfg, rec, sig, pairs_by_q)
        ctx = "\n\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(kept))
        prompt = tc.PROMPTS["gen_v3"].format(context=ctx or "(no context)",
                                             question=rec["question"])
        pred, n_in = generate(model_id, prompt)
        rows.append({
            "dataset": ds, "qid": rec["qid"], "config": cfg, "generator": model_id,
            "n_hops": rec.get("n_hops", 2), "n_kept": len(kept), "context_tokens": n_in,
            "gold_kept": sum(1 for p in kept if p["is_gold"]),
            "gold_available": sum(1 for p in rec["paragraphs"] if p["is_gold"]),
            "pred": pred, "gold": rec["answer"],
            "em": tc.exact_match(pred, rec["answer"]),
            "f1": tc.token_f1(pred, rec["answer"]),
        })
    write_jsonl(out_path, rows)
    results.commit()
    return str(out_path)


# ============================================================================
# D + E  threshold sweep and retriever generalisation
# ============================================================================

@app.function(gpu="T4", cpu=8, memory=32768, **COMMON)
def extras() -> dict:
    _env(); seed_all()
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    import triver_core as tc

    sig = pd.read_parquet(R / "adv" / "signals.parquet")

    # ---- D: no per-chunk threshold rescues the gate
    sweep = []
    for ds, d in sig.groupby("dataset"):
        for t in PC_TAUS:
            keep = d.s_ent_slot >= t
            sweep.append({
                "dataset": ds, "tau": t,
                "kappa": float(keep.mean()),
                "gold_recall": float(keep[d.is_gold].mean()),
                "answer_bearing_recall": float(keep[d.is_gold & d.answer_bearing].mean()),
                "bridge_recall": float(keep[d.is_gold & ~d.answer_bearing].mean()),
                "distractor_reject": float(1 - keep[~d.is_gold].mean()),
            })
    sw = pd.DataFrame(sweep)
    sw.to_csv(R / "results" / "analysis" / "adv_pc_sweep.csv", index=False)
    print("\n--- per-chunk threshold sweep " + "-" * 40)
    print(sw[sw.dataset == "hotpotqa"].to_string(index=False,
                                                 float_format=lambda x: f"{x:.3f}"))

    # ---- E: is the directional bias specific to one embedding model?
    recs = read_jsonl(R / "adv" / "splits" / "hotpotqa.jsonl")[:200]
    rr = []
    for mid in EMB_ALTS:
        emb = get_emb(mid)
        lab, bear, sc = [], [], []
        for rec in recs:
            na = tc.normalize_answer(rec["answer"])
            texts = [p["text"] for p in rec["paragraphs"]]
            qv = emb.encode([rec["question"]], show_progress_bar=False)[0]
            pv = emb.encode(texts, batch_size=32, show_progress_bar=False)
            cos = (pv @ qv) / (np.linalg.norm(pv, axis=1) * np.linalg.norm(qv) + 1e-9)
            for p, c in zip(rec["paragraphs"], cos):
                lab.append(int(p["is_gold"])); sc.append(float(c))
                bear.append(bool(p["is_gold"] and na not in ("yes", "no", "")
                                 and na in tc.normalize_answer(p["text"])))
        lab, sc, bear = np.array(lab), np.array(sc), np.array(bear)
        dis = lab == 0
        def a(mask):
            if mask.sum() < 20:
                return float("nan")
            return float(roc_auc_score(np.r_[np.ones(mask.sum()), np.zeros(dis.sum())],
                                       np.r_[sc[mask], sc[dis]]))
        ab, bo = a(bear), a((lab == 1) & ~bear)
        rr.append({"embedder": mid, "AUC_all": float(roc_auc_score(lab, sc)),
                   "AUC_answer_bearing": ab, "AUC_bridge_only": bo,
                   "deficit": bo - ab})
        print(f"  {mid}: deficit {bo-ab:+.3f}")
        _C.pop(f"emb::{mid}", None)
    rt = pd.DataFrame(rr)
    rt.to_csv(R / "results" / "analysis" / "adv_retriever_generalisation.csv", index=False)
    results.commit()

    print("\n" + "=" * 70)
    if (rt.deficit > 0.05).all():
        print("  The directional bias holds for every embedding model tested.")
        print("  It is a property of query-conditioned retrieval, not of bge-small.")
    else:
        print("  Deficit varies by embedder. Report per model.")
    print("=" * 70)

    with __import__("wandb").init(project="triver-rag", job_type="adv_extras",
                                  name="adv_extras", group="paper-v2", reinit=True):
        import wandb
        wandb.log({"pc_sweep": wandb.Table(dataframe=sw),
                   "retriever_gen": wandb.Table(dataframe=rt)})
    return {"sweep": sw.to_dict("records"), "retrievers": rt.to_dict("records")}


# ============================================================================
# analysis
# ============================================================================

@app.function(cpu=8, memory=32768, **COMMON)
def analyze_adv() -> dict:
    import numpy as np
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import triver_core as tc

    AN = R / "results" / "analysis"; FIG = R / "results" / "figures"
    AN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    rows = []
    for dsdir in sorted((R / "results" / "adv_gen").glob("*")):
        for mdir in sorted(dsdir.glob("*")):
            per = {}
            for cdir in sorted(mdir.glob("*")):
                rr = [r for f in sorted(cdir.glob("*.jsonl")) for r in read_jsonl(f)]
                if rr:
                    per[cdir.name] = pd.DataFrame(rr).set_index("qid").sort_index()
            for name, d in per.items():
                lo, hi = tc.wilson(int(d.em.sum()), len(d))
                rows.append({"dataset": dsdir.name, "generator": mdir.name,
                             "config": name, "n": len(d),
                             "EM": 100 * d.em.mean(), "CI_lo": 100 * lo, "CI_hi": 100 * hi,
                             "F1": d.f1.mean(),
                             "gold_recall": (d.gold_kept / d.gold_available.clip(lower=1)).mean(),
                             "ctx_tokens": d.context_tokens.mean()})
            if "B2" in per:
                ref, tests = per["B2"], []
                for name, d in per.items():
                    if name == "B2":
                        continue
                    idx = ref.index.intersection(d.index)
                    p, b, c = tc.mcnemar_exact(ref.loc[idx].em, d.loc[idx].em)
                    tests.append({"dataset": dsdir.name, "config": name, "vs": "B2",
                                  "b": b, "c": c, "p_raw": p,
                                  "em_delta": 100 * (d.loc[idx].em.mean() - ref.loc[idx].em.mean())})
                if tests:
                    for t, a in zip(tests, tc.holm([x["p_raw"] for x in tests])):
                        t["p_holm"] = a
                    pd.DataFrame(tests).to_csv(
                        AN / f"adv_tests_{dsdir.name}.csv", index=False)
    mt = pd.DataFrame(rows).sort_values(["dataset", "EM"], ascending=[True, False])
    mt.to_csv(AN / "adv_main_table.csv", index=False)
    print("\n" + mt.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # figure: hop scaling
    hp = AN / "adv_auc_by_hops.csv"
    if hp.exists():
        ht = pd.read_csv(hp)
        if len(ht):
            fig, ax = plt.subplots(figsize=(5.4, 3.4))
            for sname, g in ht.groupby("signal"):
                g = g.sort_values("n_hops")
                ax.plot(g.n_hops, g.AUC_all, marker="o", label=sname)
            ax.axhline(0.5, ls="--", c="grey", lw=1)
            ax.set_xlabel("hops required"); ax.set_ylabel("AUC, gold vs distractor")
            ax.set_xticks(sorted(ht.n_hops.unique()))
            ax.set_title("Separation degrades with hop count (MuSiQue)")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(FIG / "fig_hops.pdf")
            fig.savefig(FIG / "fig_hops.png", dpi=160)

    # figure: threshold sweep
    sp = AN / "adv_pc_sweep.csv"
    if sp.exists():
        sw = pd.read_csv(sp)
        d = sw[sw.dataset == "hotpotqa"].sort_values("tau")
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(d.tau, d.answer_bearing_recall, marker="o", label="answer-bearing gold")
        ax.plot(d.tau, d.bridge_recall, marker="s", label="bridge-only gold")
        ax.plot(d.tau, d.distractor_reject, marker="^", label="distractor rejected")
        ax.set_xlabel(r"per-chunk threshold $\tau$"); ax.set_ylabel("rate")
        ax.set_title("No threshold separates the second hop")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG / "fig_sweep.pdf")
        fig.savefig(FIG / "fig_sweep.png", dpi=160)

    # figure: end-to-end per dataset
    if len(mt):
        order = ["B1", "B2", "B4", "PC", "SET", "COND", "B5"]
        dss = sorted(mt.dataset.unique())
        fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 3.4), squeeze=False)
        for ax, ds in zip(axes[0], dss):
            d = mt[mt.dataset == ds].set_index("config").reindex(
                [c for c in order if c in set(mt.config)])
            x = np.arange(len(d))
            err = np.vstack([d.EM - d.CI_lo, d.CI_hi - d.EM])
            ax.bar(x, d.EM, 0.7, yerr=err, capsize=3, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=45, fontsize=8)
            ax.set_title(ds, fontsize=10); ax.grid(axis="y", alpha=0.3)
        axes[0][0].set_ylabel("Exact Match (%)")
        fig.tight_layout(); fig.savefig(FIG / "fig_adv_endtoend.pdf")
        fig.savefig(FIG / "fig_adv_endtoend.png", dpi=160)
    results.commit()

    with __import__("wandb").init(project="triver-rag", job_type="adv_analyze",
                                  name="adv_analyze", group="paper-v2", reinit=True):
        import wandb
        wandb.log({"adv_main": wandb.Table(dataframe=mt)})
        for f in sorted(FIG.glob("fig_*.png")):
            wandb.log({f.stem: wandb.Image(str(f))})
    return {"table": mt.to_dict("records")}


# ============================================================================

def _shards(n, size):
    return [(i, min(i + size, n)) for i in range(0, n, size)]


@app.function(cpu=2, memory=4096, **COMMON)
def available_datasets() -> list:
    d = R / "adv" / "splits"
    return sorted(f.stem for f in d.glob("*.jsonl")) if d.exists() else []


@app.local_entrypoint()
def main(stage: str = "data", shard: int = 50):
    t0 = time.time()

    if stage == "inspect":
        print(">> inspecting dataset schemas")
        inspect_datasets.remote()
        return

    if stage in ("data", "all"):
        print(">> building splits")
        meta = build_splits.remote()
        print(json.dumps(meta, indent=2, default=str))
        usable = [k for k, v in meta.items() if v.get("n", 0) > 0]
        print(f"\nusable datasets: {usable}")
        if len(usable) < 2:
            print("Fewer than two datasets built. Run --stage inspect and fix the")
            print("normaliser before continuing; the generalisation claim needs two.")

    if stage in ("signals", "all"):
        dss = available_datasets.remote()
        jobs = [(ds, lo, hi) for ds in dss for lo, hi in _shards(N_PER_DS, shard)]
        print(f">> signals: {len(jobs)} shards across {len(DATASETS)} datasets")
        list(signals_shard.starmap(jobs))

    if stage in ("auc", "signals", "all"):
        print(">> GATE 1: AUC by dataset and hop count"); auc_tables.remote()

    if stage in ("cond", "all"):
        dss = available_datasets.remote()
        jobs = [(ds, lo, hi) for ds in dss for lo, hi in _shards(N_PER_DS, shard)]
        print(f">> conditional selection: {len(jobs)} shards")
        list(cond_shard.starmap(jobs))
        print(">> GATE 2: does anchoring beat blind pair search?"); cond_gate.remote()

    if stage in ("gen", "all"):
        dss = available_datasets.remote()
        jobs = [(ds, c, lo, hi, m) for ds in dss for m in GEN_IDS
                for c in CONFIGS for lo, hi in _shards(N_PER_DS, shard)]
        print(f">> generation: {len(jobs)} shards")
        list(gen_shard.starmap(jobs))

    if stage in ("extras", "all"):
        print(">> threshold sweep and retriever generalisation"); extras.remote()

    if stage in ("gen", "extras", "analyze", "all"):
        print(">> analyze"); analyze_adv.remote()

    print(f"\nwall clock {(time.time()-t0)/60:.1f} min")
