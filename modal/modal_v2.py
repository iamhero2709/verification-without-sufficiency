"""
modal_v2.py — the four experiments from the review's section 5.2.

  8   Iterative retrieval and conditioned verification.  ~15 min, ~$2
      The reviewer's central objection: if IRCoT and Self-Ask already rewrite the
      query between hops, does the verification problem survive that fix? We test
      the retrieval side and the verification side separately.

      8a  Retrieval.  Re-score every chunk with q' = q (+) anchor, in three
          granularities (title, first sentence, full paragraph). Does the
          answer-bearing paragraph become findable?

      8b  Verification, MuSiQue only, oracle decomposition.  MuSiQue ships
          `question_decomposition` with per-hop sub-questions and the index of
          the paragraph supporting each. That gives the ceiling of what query
          rewriting could do for verification, with no heuristic rewriter.
          This is the cleanest form of the experiment.

      8c  Verification, all datasets, deployable approximation. Hypothesis
          conditioned on the anchor title rather than the gold decomposition.

  9   Two more generator sizes.  ~25 min, ~$4
      0.5B and 3B on the same seven-selector grid, so the ordering is not a
      property of one model.

  10  Answer-alias robustness.  free
      `answer_bearing` is currently exact string matching after normalisation.
      Recompute with MuSiQue aliases and a token-overlap criterion, report how
      far the split moves.

  11  Alternative generation prompt.  ~15 min, ~$2
      One prompt throughout is an obvious question. Five selectors, one
      alternative prompt.

    modal run modal_v2.py --stage rewrite    <- run this first, it is the one
    modal run modal_v2.py --stage sizes         that answers the reviewer
    modal run modal_v2.py --stage alias
    modal run modal_v2.py --stage prompt
    modal run modal_v2.py --stage analyze

Depends on artefacts written by modal_advanced.py: adv/splits/*.jsonl and
adv/signals.parquet.
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

app = modal.App("triver-v2")
R = Path("/results")

NLI_ID = "cross-encoder/nli-deberta-v3-base"
EMB_ID = "BAAI/bge-small-en-v1.5"
RERANK_ID = "BAAI/bge-reranker-base"
NEW_GEN_IDS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
DATASETS = ["hotpotqa", "2wiki", "musique"]
CONFIGS = ["B1", "B2", "B4", "PC", "SET", "COND", "B5"]
PROMPT_CONFIGS = ["B1", "B4", "PC", "COND", "B5"]
TOP_K = 5
SEED = 1337

# An alternative generation prompt for experiment 11. Differs in framing
# (extractive instruction, explicit multi-hop cue) rather than in cosmetics.
PROMPT_ALT = (
    "You are given numbered passages. Some are irrelevant. Combine information "
    "across passages if needed.\n\n{context}\n\nQuestion: {question}\n"
    "Give only the exact answer span, nothing else.\nAnswer:"
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


def get_emb():
    if "emb" not in _C:
        _env()
        import torch
        from sentence_transformers import SentenceTransformer
        _C["emb"] = SentenceTransformer(
            EMB_ID, device="cuda" if torch.cuda.is_available() else "cpu")
    return _C["emb"]


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
# 8  Iterative retrieval and conditioned verification
# ============================================================================

def first_sentence(text, cap=240):
    m = re.split(r"(?<=[.!?])\s+", text.strip())
    return (m[0] if m else text)[:cap]


def rewrite_query(q, anchor, mode):
    """q' = q (+) anchor, at three granularities.
    `title` is the Self-Ask shape: name the bridge entity.
    `sent` and `full` are the IRCoT shape: carry retrieved content forward."""
    if mode == "raw":
        return q
    if mode == "title":
        return f"{q} {anchor['title']}"
    if mode == "sent":
        return f"{q} {first_sentence(anchor['text'])}"
    if mode == "full":
        return f"{q} {anchor['text'][:600]}"
    raise ValueError(mode)


@app.function(gpu="T4", cpu=4, memory=16384, **COMMON)
def rewrite_shard(ds: str, lo: int, hi: int) -> str:
    """8a and 8c: re-score under the rewritten query, and score entailment
    under an anchor-conditioned hypothesis."""
    _env(); seed_all()
    import numpy as np
    import pandas as pd
    import triver_core as tc

    out_path = R / "v2" / "rewrite" / ds / f"{lo:05d}.jsonl"
    if out_path.exists():
        return str(out_path)

    sig = pd.read_parquet(R / "adv" / "signals.parquet")
    sig = sig[sig.dataset == ds].set_index("pid").to_dict("index")
    recs = read_jsonl(R / "adv" / "splits" / f"{ds}.jsonl")[lo:hi]
    emb = get_emb()

    rows = []
    for rec in recs:
        q, gold = rec["question"], rec["answer"]
        paras = rec["paragraphs"]
        top = sorted(paras, key=lambda p: -sig[p["pid"]]["retriever_score"])[:TOP_K]
        anchor = top[0]
        rest = top[1:]
        if not rest:
            continue
        texts = [p["text"] for p in rest]
        pv = emb.encode(texts, batch_size=32, show_progress_bar=False)

        # ---- 8a: retrieval side, four query granularities
        emb_scores = {}
        for mode in ("raw", "title", "sent", "full"):
            qv = emb.encode([rewrite_query(q, anchor, mode)],
                            show_progress_bar=False)[0]
            cos = (pv @ qv) / (np.linalg.norm(pv, axis=1) * np.linalg.norm(qv) + 1e-9)
            emb_scores[mode] = [tc.rescale_cosine(float(c)) for c in cos]

        # ---- 8c: verification side, anchor-conditioned hypothesis
        h_raw = tc.build_hypotheses(q, "prop_slot")[0]
        # A deployable approximation of a decomposed sub-question: state the
        # anchor entity as given, then ask for the remaining slot.
        h_cond = f"Given {anchor['title']}, {h_raw[0].lower()}{h_raw[1:]}"
        ent_raw = nli([(t, h_raw) for t in texts])
        ent_cond = nli([(t, h_cond) for t in texts])

        na = tc.normalize_answer(gold)
        for j, p in enumerate(rest):
            rows.append({
                "dataset": ds, "qid": rec["qid"], "pid": p["pid"],
                "n_hops": rec.get("n_hops", 2), "qtype": rec.get("qtype", "?"),
                "is_gold": p["is_gold"],
                "answer_bearing": bool(na not in ("yes", "no", "")
                                       and na in tc.normalize_answer(p["text"])),
                "anchor_is_gold": bool(anchor["is_gold"]),
                "s_emb_raw": emb_scores["raw"][j],
                "s_emb_title": emb_scores["title"][j],
                "s_emb_sent": emb_scores["sent"][j],
                "s_emb_full": emb_scores["full"][j],
                "s_ent_raw": ent_raw[j],
                "s_ent_cond": ent_cond[j],
            })
    write_jsonl(out_path, rows)
    results.commit()
    return str(out_path)


@app.function(gpu="T4", cpu=4, memory=16384, **COMMON)
def musique_decomposition(n_q: int = 500) -> dict:
    """8b: the clean version. MuSiQue ships per-hop sub-questions and the index
    of the paragraph supporting each hop, so we can measure what a perfect query
    rewriter would buy a verifier, without building one."""
    _env(); seed_all()
    import numpy as np
    from datasets import load_dataset
    from sklearn.metrics import roc_auc_score
    import triver_core as tc

    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    keep = {r["qid"] for r in read_jsonl(R / "adv" / "splits" / "musique.jsonl")}
    ex_by_id = {str(e["id"]): e for e in ds if str(e["id"]) in keep}
    print(f"matched {len(ex_by_id)} MuSiQue records to the split")

    def resolve(text, prev_answers):
        """Sub-questions carry #1, #2 placeholders for earlier hop answers."""
        for k, a in prev_answers.items():
            text = text.replace(f"#{k}", str(a))
        return text

    pairs_orig, pairs_sub, pairs_sub_oracle, labels, hops = [], [], [], [], []
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
                continue                      # hop 1 is the easy one, skip it
            # MuSiQue sub-questions are relational: "X >> relation"
            if ">>" in sub:
                subj, rel = [s.strip() for s in sub.split(">>", 1)]
            else:
                subj, rel = sub, "answer"
            h_sub = f"The {rel} of {subj} is something."
            h_sub_or = f"The {rel} of {subj} is {step.get('answer','')}."
            h_orig = tc.build_hypotheses(ex["question"], "prop_slot")[0]
            for j, p in enumerate(paras):
                t = (str(p.get("title", "")) + ". " +
                     str(p.get("paragraph_text", ""))).strip()
                pairs_orig.append((t, h_orig))
                pairs_sub.append((t, h_sub))
                pairs_sub_oracle.append((t, h_sub_or))
                labels.append(int(int(p.get("idx", j)) == int(idx)))
                hops.append(len(decomp))
            break                              # one later hop per question

    print(f"scoring {len(pairs_orig)} pairs x 3 hypothesis forms")
    s_orig = nli(pairs_orig)
    s_sub = nli(pairs_sub)
    s_sub_or = nli(pairs_sub_oracle)

    y = np.array(labels)
    out = {
        "n_pairs": len(y), "n_positive": int(y.sum()),
        "AUC_original_question": round(float(roc_auc_score(y, s_orig)), 4),
        "AUC_decomposed_subquestion": round(float(roc_auc_score(y, s_sub)), 4),
        "AUC_decomposed_with_answer": round(float(roc_auc_score(y, s_sub_or)), 4),
    }
    out["lift_from_decomposition"] = round(
        out["AUC_decomposed_subquestion"] - out["AUC_original_question"], 4)
    # per hop count
    h = np.array(hops)
    for k in sorted(set(hops)):
        m = h == k
        if m.sum() > 100 and 0 < y[m].sum() < m.sum():
            out[f"AUC_original_{k}hop"] = round(
                float(roc_auc_score(y[m], np.array(s_orig)[m])), 4)
            out[f"AUC_decomposed_{k}hop"] = round(
                float(roc_auc_score(y[m], np.array(s_sub)[m])), 4)

    d = R / "results" / "analysis"; d.mkdir(parents=True, exist_ok=True)
    (d / "v2_musique_decomposition.json").write_text(json.dumps(out, indent=2))
    results.commit()

    print("\n" + json.dumps(out, indent=2))
    print("\n" + "=" * 70)
    if out["lift_from_decomposition"] >= 0.15:
        print("  DECOMPOSITION FIXES VERIFICATION. Entailment on the later hop")
        print(f"  rises from {out['AUC_original_question']:.3f} to "
              f"{out['AUC_decomposed_subquestion']:.3f} when the hypothesis is")
        print("  built from the sub-question instead of the original query.")
        print("  The paper gains a positive result: verification must be")
        print("  conditioned on the same rewritten query retrieval uses.")
    elif out["lift_from_decomposition"] >= 0.05:
        print("  Partial lift. Report it and keep the claim narrow.")
    else:
        print("  Decomposition does not rescue verification either. That is a")
        print("  stronger negative result: even a perfect query rewriter, which")
        print("  is what IRCoT approximates, leaves the verifier blind.")
    print("=" * 70)

    with __import__("wandb").init(project="triver-rag", job_type="v2_decomp",
                                  name="v2_decomp", group="paper-v2", reinit=True):
        import wandb; wandb.log(out)
    return out


@app.function(cpu=8, memory=32768, **COMMON)
def rewrite_analysis() -> dict:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    frames = []
    for ds in DATASETS:
        d = R / "v2" / "rewrite" / ds
        if d.exists():
            frames.append(pd.DataFrame([r for f in sorted(d.glob("*.jsonl"))
                                        for r in read_jsonl(f)]))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(R / "v2" / "rewrite.parquet")

    cols = ["s_emb_raw", "s_emb_title", "s_emb_sent", "s_emb_full",
            "s_ent_raw", "s_ent_cond"]
    rows = []
    for ds, d in df.groupby("dataset"):
        dis = d[~d.is_gold]
        bear = d[d.is_gold & d.answer_bearing]
        for c in cols:
            def auc(pos):
                if len(pos) < 20:
                    return float("nan")
                return float(roc_auc_score([1] * len(pos) + [0] * len(dis),
                                           list(pos[c]) + list(dis[c])))
            rows.append({"dataset": ds, "signal": c,
                         "AUC_all_gold": float(roc_auc_score(d.is_gold, d[c])),
                         "AUC_answer_bearing": auc(bear)})
    t = pd.DataFrame(rows)
    base = t[t.signal == "s_emb_raw"].set_index("dataset").AUC_answer_bearing
    t["lift_vs_raw"] = t.apply(
        lambda r: r.AUC_answer_bearing - base.get(r.dataset, np.nan)
        if r.signal.startswith("s_emb") else np.nan, axis=1)
    ebase = t[t.signal == "s_ent_raw"].set_index("dataset").AUC_answer_bearing
    t.loc[t.signal == "s_ent_cond", "lift_vs_raw"] = t[t.signal == "s_ent_cond"].apply(
        lambda r: r.AUC_answer_bearing - ebase.get(r.dataset, np.nan), axis=1)

    t.to_csv(R / "results" / "analysis" / "v2_rewrite.csv", index=False)
    results.commit()
    print("\n--- retrieval and verification under a rewritten query " + "-" * 14)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best = t[(t.signal == "s_emb_full")].lift_vs_raw.mean()
    print("\n" + "=" * 70)
    print(f"  mean lift on answer-bearing gold, full-paragraph rewrite: {best:+.3f}")
    if best >= 0.05:
        print("  Query rewriting recovers the second hop for RETRIEVAL.")
        print("  Whether it also fixes verification is the s_ent_cond row and")
        print("  the MuSiQue decomposition experiment.")
    else:
        print("  Rewriting does not recover the second hop here either.")
    print("=" * 70)

    with __import__("wandb").init(project="triver-rag", job_type="v2_rewrite",
                                  name="v2_rewrite", group="paper-v2", reinit=True):
        import wandb; wandb.log({"rewrite": wandb.Table(dataframe=t)})
    return {"table": t.to_dict("records")}


# ============================================================================
# 9 and 11  generation: more sizes, and one alternative prompt
# ============================================================================

def select(cfg, rec, sig, pairs_by_q):
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
        keep = [p for p in top if sig[p["pid"]]["s_ent_slot"] >= 0.35]
        return keep if keep else top[:1]
    r = pairs_by_q.get(rec["qid"])
    if r is None:
        return top[:2]
    key = "set_pids" if cfg == "SET" else "cond_pids"
    return [by_pid[r[key][0]], by_pid[r[key][1]]]


@app.function(gpu="L4", cpu=4, memory=32768, **COMMON)
def gen_shard(ds: str, cfg: str, lo: int, hi: int, model_id: str,
              prompt_id: str = "gen_v3") -> str:
    _env(); seed_all()
    import pandas as pd
    import triver_core as tc

    tag = model_id.split("/")[-1]
    out_path = R / "v2" / "gen" / prompt_id / ds / tag / cfg / f"{lo:05d}.jsonl"
    if out_path.exists():
        return str(out_path)

    sig = pd.read_parquet(R / "adv" / "signals.parquet")
    sig = sig[sig.dataset == ds].set_index("pid").to_dict("index")
    cdf = pd.read_parquet(R / "adv" / "cond.parquet")
    pairs_by_q = {r["qid"]: r for r in cdf[cdf.dataset == ds].to_dict("records")}
    recs = read_jsonl(R / "adv" / "splits" / f"{ds}.jsonl")[lo:hi]
    recs = [r for r in recs if int(r.get("n_hops", 2)) == 2]
    if not recs:
        write_jsonl(out_path, []); results.commit(); return str(out_path)

    template = PROMPT_ALT if prompt_id == "gen_alt" else tc.PROMPTS["gen_v3"]
    rows = []
    for rec in recs:
        kept = select(cfg, rec, sig, pairs_by_q)
        ctx = "\n\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(kept))
        prompt = template.format(context=ctx or "(no context)",
                                 question=rec["question"])
        pred, n_in = generate(model_id, prompt)
        rows.append({
            "dataset": ds, "qid": rec["qid"], "config": cfg, "generator": model_id,
            "prompt_id": prompt_id, "n_kept": len(kept), "context_tokens": n_in,
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
# 10  answer-alias robustness
# ============================================================================

@app.function(cpu=8, memory=32768, **COMMON)
def alias_robustness() -> dict:
    """`answer_bearing` currently uses exact normalised string matching, which
    misses paraphrases and aliases. Recompute three ways and report how far the
    central split moves."""
    _env()
    import numpy as np
    import pandas as pd
    from collections import Counter
    from datasets import load_dataset
    from sklearn.metrics import roc_auc_score
    import triver_core as tc

    sig = pd.read_parquet(R / "adv" / "signals.parquet")

    text_of, ans_of, alias_of = {}, {}, {}
    for ds in DATASETS:
        p = R / "adv" / "splits" / f"{ds}.jsonl"
        if not p.exists():
            continue
        for rec in read_jsonl(p):
            for par in rec["paragraphs"]:
                text_of[par["pid"]] = par["text"]
                ans_of[par["pid"]] = rec["answer"]
    try:
        mq = load_dataset("dgslibisey/MuSiQue", split="validation")
        al = {str(e["id"]): list(e.get("answer_aliases") or []) for e in mq}
        for pid in list(text_of):
            qid = pid.split("::")[0]
            if qid in al:
                alias_of[pid] = al[qid]
    except Exception as e:
        print(f"aliases unavailable: {type(e).__name__}: {e}")

    df = sig.copy()
    df["text"] = df.pid.map(text_of)
    df["answer"] = df.pid.map(ans_of)
    df = df.dropna(subset=["text", "answer"])

    def exact(r):
        a = tc.normalize_answer(str(r["answer"]))
        return a not in ("yes", "no", "") and a in tc.normalize_answer(str(r["text"]))

    def with_alias(r):
        if exact(r):
            return True
        for a in alias_of.get(r["pid"], []):
            na = tc.normalize_answer(str(a))
            if na and na not in ("yes", "no") and na in tc.normalize_answer(str(r["text"])):
                return True
        return False

    def token_overlap(r, thresh=0.8):
        a = tc.normalize_answer(str(r["answer"])).split()
        if not a or " ".join(a) in ("yes", "no"):
            return False
        t = Counter(tc.normalize_answer(str(r["text"])).split())
        hit = sum(min(c, t[w]) for w, c in Counter(a).items())
        return hit / len(a) >= thresh

    df["ab_exact"] = df.apply(exact, axis=1)
    df["ab_alias"] = df.apply(with_alias, axis=1)
    df["ab_overlap"] = df.apply(token_overlap, axis=1)

    rows = []
    for ds, d in df.groupby("dataset"):
        dis = d[~d.is_gold]
        for crit in ("ab_exact", "ab_alias", "ab_overlap"):
            bear = d[d.is_gold & d[crit]]
            brid = d[d.is_gold & ~d[crit]]
            def auc(pos, col):
                if len(pos) < 20:
                    return float("nan")
                return float(roc_auc_score([1] * len(pos) + [0] * len(dis),
                                           list(pos[col]) + list(dis[col])))
            rows.append({
                "dataset": ds, "criterion": crit,
                "frac_gold_answer_bearing": float(d[d.is_gold][crit].mean()),
                "emb_answer_bearing": auc(bear, "s_emb"),
                "emb_bridge_only": auc(brid, "s_emb"),
                "emb_deficit": auc(brid, "s_emb") - auc(bear, "s_emb"),
                "ent_answer_bearing": auc(bear, "s_ent_slot"),
                "ent_bridge_only": auc(brid, "s_ent_slot"),
            })
    t = pd.DataFrame(rows)
    t.to_csv(R / "results" / "analysis" / "v2_alias_robustness.csv", index=False)
    results.commit()
    print("\n--- answer-bearing definition sensitivity " + "-" * 28)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    spread = t.groupby("dataset").emb_deficit.agg(lambda s: s.max() - s.min())
    print("\n" + "=" * 70)
    print("  deficit spread across the three definitions:")
    for k, v in spread.items():
        print(f"    {k:9s} {v:.3f}")
    print("  If the spread is small the split is robust and the finding does")
    print("  not depend on how answer-bearing is operationalised.")
    print("=" * 70)
    return {"table": t.to_dict("records")}


# ============================================================================
# analysis
# ============================================================================

@app.function(cpu=8, memory=32768, **COMMON)
def analyze_v2() -> dict:
    import numpy as np
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import triver_core as tc

    AN = R / "results" / "analysis"; FIG = R / "results" / "figures"
    AN.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    rows, tests = [], []
    root = R / "v2" / "gen"
    for pdir in sorted(root.glob("*")):
        for dsdir in sorted(pdir.glob("*")):
            for mdir in sorted(dsdir.glob("*")):
                per = {}
                for cdir in sorted(mdir.glob("*")):
                    rr = [r for f in sorted(cdir.glob("*.jsonl")) for r in read_jsonl(f)]
                    if rr:
                        per[cdir.name] = pd.DataFrame(rr).set_index("qid").sort_index()
                for name, d in per.items():
                    lo, hi = tc.wilson(int(d.em.sum()), len(d))
                    rows.append({"prompt": pdir.name, "dataset": dsdir.name,
                                 "generator": mdir.name, "config": name, "n": len(d),
                                 "EM": 100 * d.em.mean(), "CI_lo": 100 * lo,
                                 "CI_hi": 100 * hi, "F1": d.f1.mean(),
                                 "gold_recall": (d.gold_kept /
                                                 d.gold_available.clip(lower=1)).mean()})
                if "B1" in per:
                    ref = per["B1"]
                    for name, d in per.items():
                        if name == "B1":
                            continue
                        idx = ref.index.intersection(d.index)
                        p, b, c = tc.mcnemar_exact(ref.loc[idx].em, d.loc[idx].em)
                        tests.append({"prompt": pdir.name, "dataset": dsdir.name,
                                      "generator": mdir.name, "config": name,
                                      "vs": "B1", "b": b, "c": c, "p_raw": p,
                                      "em_delta": 100 * (d.loc[idx].em.mean()
                                                         - ref.loc[idx].em.mean())})
    mt = pd.DataFrame(rows)
    if len(mt):
        mt = mt.sort_values(["prompt", "dataset", "generator", "EM"],
                            ascending=[True, True, True, False])
        mt.to_csv(AN / "v2_main_table.csv", index=False)
        print("\n" + mt.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if tests:
        td = pd.DataFrame(tests)
        for key, g in td.groupby(["prompt", "dataset", "generator"]):
            td.loc[g.index, "p_holm"] = tc.holm(list(g.p_raw))
        td.to_csv(AN / "v2_pairwise_tests.csv", index=False)

    # figure: selector ordering across generator sizes
    base = mt[mt.prompt == "gen_v3"] if len(mt) else mt
    if len(base):
        order = ["B1", "B2", "B4", "PC", "SET", "COND", "B5"]
        gens = sorted(base.generator.unique(),
                      key=lambda s: float(re.search(r"([\d.]+)B", s).group(1))
                      if re.search(r"([\d.]+)B", s) else 0)
        dss = sorted(base.dataset.unique())
        fig, axes = plt.subplots(1, len(dss), figsize=(3.7 * len(dss), 3.6),
                                 squeeze=False)
        for ax, ds in zip(axes[0], dss):
            for g in gens:
                d = base[(base.dataset == ds) & (base.generator == g)] \
                    .set_index("config").reindex([c for c in order
                                                  if c in set(base.config)])
                ax.plot(range(len(d)), d.EM, marker="o", ms=4, label=g)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, fontsize=8)
            ax.set_title(ds, fontsize=10); ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("Exact Match (%)")
        axes[0][-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "fig_sizes.pdf"); fig.savefig(FIG / "fig_sizes.png", dpi=160)
    results.commit()

    with __import__("wandb").init(project="triver-rag", job_type="v2_analyze",
                                  name="v2_analyze", group="paper-v2", reinit=True):
        import wandb
        if len(mt):
            wandb.log({"v2_main": wandb.Table(dataframe=mt)})
        for f in sorted(FIG.glob("fig_sizes.png")):
            wandb.log({f.stem: wandb.Image(str(f))})
    return {"n_rows": len(mt)}


# ============================================================================

def _shards(n, size):
    return [(i, min(i + size, n)) for i in range(0, n, size)]


@app.local_entrypoint()
def main(stage: str = "rewrite", n: int = 500, shard: int = 50):
    t0 = time.time()

    if stage in ("rewrite", "all"):
        jobs = [(ds, lo, hi) for ds in DATASETS for lo, hi in _shards(n, shard)]
        print(f">> 8a/8c rewrite: {len(jobs)} shards")
        list(rewrite_shard.starmap(jobs))
        rewrite_analysis.remote()
        print(">> 8b MuSiQue decomposition (the clean version)")
        musique_decomposition.remote()

    if stage in ("sizes", "all"):
        jobs = [(ds, c, lo, hi, m, "gen_v3") for ds in DATASETS
                for m in NEW_GEN_IDS for c in CONFIGS for lo, hi in _shards(n, shard)]
        print(f">> 9 generator sizes: {len(jobs)} shards")
        list(gen_shard.starmap(jobs))

    if stage in ("prompt", "all"):
        jobs = [(ds, c, lo, hi, "Qwen/Qwen2.5-1.5B-Instruct", "gen_alt")
                for ds in DATASETS for c in PROMPT_CONFIGS
                for lo, hi in _shards(n, shard)]
        print(f">> 11 alternative prompt: {len(jobs)} shards")
        list(gen_shard.starmap(jobs))

    if stage in ("alias", "all"):
        print(">> 10 answer-alias robustness"); alias_robustness.remote()

    if stage in ("sizes", "prompt", "analyze", "all"):
        print(">> analyze"); analyze_v2.remote()

    print(f"\nwall clock {(time.time()-t0)/60:.1f} min")
