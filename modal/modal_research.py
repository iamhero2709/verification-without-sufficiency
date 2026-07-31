"""
modal_research.py — the whole research pipeline for the TriVer-RAG paper.

    modal run modal_research.py --stage gate      # E1-E4  ~15 min, ~$1     <- run this first
    modal run modal_research.py --stage calib     # E5     ~1 h,   ~$2
    modal run modal_research.py --stage main      # E6     ~1 h,   ~$4.5
    modal run modal_research.py --stage scale     # E7     ~9 h,   ~$10
    modal run modal_research.py --stage analyze   # E10 + all figures/tables
    modal run modal_research.py --stage all

E8 (cost-model fit) and E9 (latency) CANNOT run here. They measure the edge
device, which is the paper's premise. The analyze stage writes LOCAL_RUNS.md
with the exact commands; drop the resulting JSONL into the results volume and
rerun analyze to pick them up.

One-time setup
--------------
    pip install modal && modal setup
    modal secret create huggingface-secret HF_TOKEN=hf_xxx
    modal secret create wandb-secret WANDB_API_KEY=xxx

Modal renames decorator kwargs between releases (keep_warm -> min_containers,
container_idle_timeout -> scaledown_window). If a kwarg below is rejected,
check modal.com/docs rather than guessing.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

# ============================================================================
# App, image, volumes, secrets
# ============================================================================

APP_NAME = "triver-rag"
WANDB_PROJECT = "triver-rag"

SEED = 1337
N_CALIB, N_EVAL, N_TIMING = 200, 500, 60
EMB_ID = "BAAI/bge-small-en-v1.5"
NLI_ID = "cross-encoder/nli-deberta-v3-xsmall"
NLI_PROBE_IDS = [
    "cross-encoder/nli-deberta-v3-xsmall",          # 44M, the edge candidate
    "cross-encoder/nli-deberta-v3-base",            # 184M
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",  # 435M
]
GEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SCALE_IDS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

SMALL_GPU = "T4"      # signals, NLI probe
GEN_GPU = "L4"        # generation; "A10G" if L4 is unavailable
REGION = "us-east"    # pin it: the regional multiplier ranges 1.25x to 2.5x

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "tokenizers==0.19.1",
        "sentencepiece==0.2.0",
        "sentence-transformers==3.0.1",
        "accelerate==0.33.0",
        "datasets==2.21.0",
        "spacy==3.7.5",
        "numpy==1.26.4",
        "scipy==1.14.0",
        "scikit-learn==1.5.1",
        "pandas==2.2.2",
        "pyarrow==17.0.0",
        "matplotlib==3.9.2",
        "wandb==0.17.7",
    )
    .run_commands("python -m spacy download en_core_web_sm")
    .add_local_python_source("triver_core")
)

cache = modal.Volume.from_name("triver-cache", create_if_missing=True)     # HF weights + datasets
results = modal.Volume.from_name("triver-results", create_if_missing=True)  # everything the paper needs

VOLS = {"/cache": cache, "/results": results}
SECRETS = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("wandb-secret"),
]
COMMON = dict(image=image, volumes=VOLS, secrets=SECRETS,
              timeout=60 * 60 * 2, retries=2, region=REGION)

app = modal.App(APP_NAME)

R = Path("/results")


# ============================================================================
# Container-local helpers
# ============================================================================

_CACHED = {}


def _env():
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("HF_DATASETS_CACHE", "/cache/hf/datasets")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_all(seed: int = SEED):
    import random
    import numpy as np
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_embedder():
    if "emb" not in _CACHED:
        _env()
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.perf_counter()
        _CACHED["emb"] = SentenceTransformer(EMB_ID, device=dev)
        print(f"[load] embedder in {time.perf_counter()-t0:.1f}s on {dev}")
    return _CACHED["emb"]


def get_nli(model_id: str = NLI_ID):
    key = f"nli::{model_id}"
    if key not in _CACHED:
        _env()
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_id).to(dev).eval()
        _CACHED[key] = (tok, mdl, dev)
        print(f"[load] {model_id} in {time.perf_counter()-t0:.1f}s on {dev}")
    return _CACHED[key]


def get_generator(model_id: str = GEN_ID):
    key = f"gen::{model_id}"
    if key not in _CACHED:
        _env()
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=None).to(dev).eval()
        _CACHED[key] = (tok, mdl, dev)
        print(f"[load] {model_id} in {time.perf_counter()-t0:.1f}s on {dev} ({dtype})")
    return _CACHED[key]


def get_spacy():
    if "nlp" not in _CACHED:
        import spacy
        _CACHED["nlp"] = spacy.load("en_core_web_sm")
    return _CACHED["nlp"]


def wandb_run(job: str, cfg: dict | None = None):
    import wandb
    return wandb.init(project=WANDB_PROJECT, job_type=job, name=job,
                      group="paper-v1", config=cfg or {}, reinit=True)


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ============================================================================
# Stage 0 — prepare: dataset, splits, whitener
# ============================================================================

@app.function(cpu=4, memory=16384, **COMMON)
def prepare() -> dict:
    """Download HotpotQA, materialise the three disjoint splits, fit the whitener."""
    _env(); seed_all()
    import numpy as np
    from datasets import load_dataset
    import triver_core as tc

    (R / "splits").mkdir(parents=True, exist_ok=True)

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    print(f"HotpotQA dev: {len(ds)} questions")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(ds))
    calib = idx[:N_CALIB]
    eval_ = idx[N_CALIB:N_CALIB + N_EVAL]
    timing = eval_[:N_TIMING]           # timing is a SUBSET of eval, by design

    # Flatten to a compact record: 10 paragraphs per question in the distractor
    # setting, so retrieval is per-question and no global index is needed.
    def pack(i):
        ex = ds[int(i)]
        titles = ex["context"]["title"]
        sents = ex["context"]["sentences"]
        gold_titles = set(ex["supporting_facts"]["title"])
        return {
            "qid": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "level": ex["level"],
            "paragraphs": [
                {"pid": f'{ex["id"]}::{j}', "title": t,
                 "text": (t + ". " + " ".join(s)).strip(),
                 "is_gold": bool(t in gold_titles)}
                for j, (t, s) in enumerate(zip(titles, sents))
            ],
        }

    for name, ids in [("calib", calib), ("eval", eval_), ("timing", timing)]:
        rows = [pack(i) for i in ids]
        write_jsonl(R / "splits" / f"{name}.jsonl", rows)
        with open(R / "splits" / f"{name}.txt", "w") as f:
            f.write("\n".join(r["qid"] for r in rows))
        print(f"  {name}: {len(rows)}")

    cs = {r["qid"] for r in read_jsonl(R / "splits" / "calib.jsonl")}
    es = {r["qid"] for r in read_jsonl(R / "splits" / "eval.jsonl")}
    assert not (cs & es), "calib and eval overlap; splits are invalid"

    # Fit the whitener on filler embeddings drawn from the corpus.
    emb = get_embedder(); nlp = get_spacy()
    fillers = []
    for r in read_jsonl(R / "splits" / "calib.jsonl"):
        for p in r["paragraphs"]:
            doc = nlp(p["text"][:1000])
            fillers.extend(list(tc.extract_roles(doc).values()))
            fillers.extend([e.text for e in doc.ents][:5])
        if len(fillers) > 20000:
            break
    fillers = list(dict.fromkeys(fillers))[:20000]
    X = np.asarray(emb.encode(fillers, batch_size=256, show_progress_bar=False))
    wh = tc.Whitener().fit(X)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    wh.save(R / "artifacts" / "whitener.npz")

    check = tc.Whitener.verify(X, wh)
    print(f"[whitening gate] {check}")

    meta = {"n_calib": len(cs), "n_eval": len(es), "n_fillers": len(fillers), **check}
    (R / "artifacts" / "prepare.json").write_text(json.dumps(meta, indent=2))
    results.commit()

    with wandb_run("prepare", meta):
        import wandb; wandb.log(meta)
    return meta


# ============================================================================
# Stage 1 — E1..E4: signals, AUC, NLI ceiling, NLI capacity, HRR 2x2
# ============================================================================

def _nli_entail(pairs, model_id):
    """pairs: list of (premise, hypothesis). Returns P(entailment) each."""
    import torch
    tok, mdl, dev = get_nli(model_id)
    lab = {k.lower(): v for k, v in mdl.config.label2id.items()}
    ent_ix = lab.get("entailment", 0)
    out = []
    B = 32
    with torch.inference_mode():
        for i in range(0, len(pairs), B):
            batch = pairs[i:i + B]
            enc = tok([p for p, _ in batch], [h for _, h in batch],
                      return_tensors="pt", padding=True, truncation=True,
                      max_length=384).to(dev)
            probs = torch.softmax(mdl(**enc).logits, dim=-1)
            out.extend(probs[:, ent_ix].float().cpu().tolist())
    return out


@app.function(gpu=SMALL_GPU, cpu=4, memory=16384, **COMMON)
def signals_shard(split: str, lo: int, hi: int) -> str:
    """Score every (question, paragraph) pair on every signal variant."""
    _env(); seed_all()
    import numpy as np
    import triver_core as tc

    out_path = R / "artifacts" / "signals" / f"{split}_{lo:05d}.jsonl"
    if out_path.exists():
        return str(out_path)                      # idempotent: preemption is free

    rows_in = read_jsonl(R / "splits" / f"{split}.jsonl")[lo:hi]
    emb = get_embedder(); nlp = get_spacy()
    wh = tc.Whitener.load(R / "artifacts" / "whitener.npz")
    rvecs = tc.role_vectors(384, seed=SEED)
    embed_one = lambda s: emb.encode([s], show_progress_bar=False)[0]

    out = []
    for rec in rows_in:
        q, gold = rec["question"], rec["answer"]
        paras = rec["paragraphs"]
        texts = [p["text"] for p in paras]

        qv = emb.encode([q], show_progress_bar=False)[0]
        pv = emb.encode(texts, batch_size=32, show_progress_bar=False)
        cos = (pv @ qv) / (np.linalg.norm(pv, axis=1) * np.linalg.norm(qv) + 1e-9)

        qdoc = nlp(q)
        q_roles = tc.extract_roles(qdoc)
        docs = [nlp(t[:2000]) for t in texts]

        # ---- NLI, three hypothesis modes (this is what E2 gates on)
        hyp_slot = tc.build_hypotheses(q, "prop_slot")[0]
        hyp_orac = tc.build_hypotheses(q, "prop_oracle", gold=gold)[0]
        ent_slot = _nli_entail([(t, hyp_slot) for t in texts], NLI_ID)
        ent_orac = _nli_entail([(t, hyp_orac) for t in texts], NLI_ID)
        ent_raw = _nli_entail([(t, q) for t in texts], NLI_ID)
        ent_ans = []
        for t, d in zip(texts, docs):
            hyps = tc.build_hypotheses(q, "prop_answer", chunk_doc=d)
            scores = _nli_entail([(t, h) for h in hyps], NLI_ID)
            ent_ans.append(max(scores) if scores else 0.0)

        # ---- HRR 2x2: {chunk, sentence} x {raw, whitened}
        ident = tc.Whitener()          # identity + L2 norm = "raw fillers"
        hrr = {}
        for gname in ("chunk", "sentence"):
            for wname, W in (("raw", ident), ("white", wh)):
                vals, ks = [], []
                for t, d in zip(texts, docs):
                    if gname == "chunk":
                        units = [d]
                    else:
                        units = list(d.sents) if d.has_annotation("SENT_START") else [d]
                    traces = []
                    for u in units:
                        tr, k = tc.build_trace(tc.extract_roles(u), embed_one, rvecs, W)
                        traces.append(tr); ks.append(k)
                    vals.append(tc.hrr_score(q_roles, traces, embed_one, rvecs, W, agg="max"))
                hrr[f"{gname}_{wname}"] = vals
                hrr[f"k_{gname}"] = ks[:len(texts)] if ks else [0] * len(texts)

        for j, p in enumerate(paras):
            out.append({
                "qid": rec["qid"], "pid": p["pid"], "title": p["title"],
                "is_gold": p["is_gold"], "retriever_rank": int(np.argsort(-cos).tolist().index(j)),
                "retriever_score": float(cos[j]),
                "s_emb": tc.rescale_cosine(float(cos[j])),
                "s_ent_slot": ent_slot[j], "s_ent_oracle": ent_orac[j],
                "s_ent_raw": ent_raw[j], "s_ent_answer": ent_ans[j],
                "s_str_chunk_raw": hrr["chunk_raw"][j],
                "s_str_chunk_white": hrr["chunk_white"][j],
                "s_str_sent_raw": hrr["sentence_raw"][j],
                "s_str_sent_white": hrr["sentence_white"][j],
                "k_chunk": hrr["k_chunk"][j] if j < len(hrr["k_chunk"]) else 0,
                "chunk_tokens": len(p["text"].split()),
            })

    write_jsonl(out_path, out)
    results.commit()
    return str(out_path)


@app.function(gpu=SMALL_GPU, cpu=4, memory=16384, **COMMON)
def nli_probe(n_pairs: int = 200) -> dict:
    """E3 — is 44M too small, or is the approach wrong?"""
    _env(); seed_all()
    import numpy as np
    import triver_core as tc
    from sklearn.metrics import roc_auc_score

    recs = read_jsonl(R / "splits" / "calib.jsonl")
    pairs, labels = [], []
    for rec in recs:
        for p in rec["paragraphs"]:
            pairs.append((rec["question"], rec["answer"], p["text"]))
            labels.append(int(p["is_gold"]))
            if len(pairs) >= n_pairs:
                break
        if len(pairs) >= n_pairs:
            break

    table = {}
    for mid in NLI_PROBE_IDS:
        row = {}
        for mode in ("prop_slot", "prop_oracle"):
            inp = []
            for q, g, t in pairs:
                h = tc.build_hypotheses(q, mode, gold=g)[0]
                inp.append((t, h))
            sc = _nli_entail(inp, mid)
            row[mode] = float(roc_auc_score(labels, sc))
        table[mid] = row
        print(f"{mid}: {row}")
        _CACHED.pop(f"nli::{mid}", None)      # free VRAM before the next model

    (R / "results" / "analysis").mkdir(parents=True, exist_ok=True)
    (R / "results" / "analysis" / "e3_nli_probe.json").write_text(json.dumps(table, indent=2))
    results.commit()
    with wandb_run("e3_nli_probe"):
        import wandb; wandb.log({"nli_probe": table})
    return table


@app.function(cpu=8, memory=16384, **COMMON)
def signal_auc() -> dict:
    """E1, E2, E4 — the AUC table, the ceiling gate, the HRR 2x2."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    files = sorted((R / "artifacts" / "signals").glob("*.jsonl"))
    df = pd.DataFrame([r for f in files for r in read_jsonl(f)])
    df.to_parquet(R / "artifacts" / "signals.parquet")

    cols = [c for c in df.columns if c.startswith(("s_emb", "s_ent", "s_str"))]
    auc = {c: float(roc_auc_score(df["is_gold"], df[c])) for c in cols}
    k_stats = {"k_chunk_mean": float(df["k_chunk"].mean()),
               "k_chunk_p50": float(df["k_chunk"].median()),
               "k_chunk_p90": float(df["k_chunk"].quantile(0.9))}

    ceiling = auc.get("s_ent_oracle", 0.0)
    gate = ("PROCEED" if ceiling >= 0.85 else
            "PROBE_FIRST" if ceiling >= 0.70 else "NEGATIVE_RESULT_PAPER")

    out = {"auc": auc, "k": k_stats, "n_pairs": len(df),
           "E2_oracle_ceiling": ceiling, "E2_gate": gate}
    (R / "results" / "analysis").mkdir(parents=True, exist_ok=True)
    (R / "results" / "analysis" / "e1_signal_auc.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame([{"signal": k, "auc": v} for k, v in auc.items()]).to_csv(
        R / "results" / "analysis" / "signal_auc.csv", index=False)
    results.commit()

    print("\n" + "=" * 64)
    for k, v in sorted(auc.items(), key=lambda x: -x[1]):
        print(f"  {k:24s} AUC {v:.3f}")
    print(f"\n  E2 ORACLE CEILING = {ceiling:.3f}  ->  {gate}")
    print("=" * 64 + "\n")

    with wandb_run("e1_signal_auc", out):
        import wandb
        wandb.log({f"auc/{k}": v for k, v in auc.items()})
        wandb.log({"E2_oracle_ceiling": ceiling, **k_stats})
    return out


# ============================================================================
# Stage 2/3 — E5 calibration and E6 main ablation
# ============================================================================

def _generate(model_id, prompt, max_new_tokens=32):
    import torch
    tok, mdl, dev = get_generator(model_id)
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok([text], return_tensors="pt").to(dev)
    with torch.inference_mode():
        out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           num_beams=1, pad_token_id=tok.eos_token_id)
    gen = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip(), int(enc["input_ids"].shape[1])


def _run_one(rec, sig_by_pid, cfg, model_id):
    """THE function. The local latency runner must call this same code."""
    import numpy as np
    import triver_core as tc

    q, gold = rec["question"], rec["answer"]
    paras = rec["paragraphs"]

    t0 = time.perf_counter()
    if cfg.retriever == "none":
        cands = []
    elif cfg.retriever == "oracle":
        cands = [p for p in paras if p["is_gold"]]
    else:
        ranked = sorted(paras, key=lambda p: -sig_by_pid[p["pid"]]["retriever_score"])
        cands = ranked[:cfg.top_k] if cfg.top_k else ranked
    t_ret = time.perf_counter() - t0

    t0 = time.perf_counter()
    retained, verdicts, fused = [], {}, {}
    if cfg.use_emb or cfg.use_nli or cfg.use_hrr:
        for p in cands:
            s = sig_by_pid[p["pid"]]
            sig = tc.Signals(
                s_emb=s["s_emb"],
                s_str=s.get(f"s_str_{cfg.hrr_granularity}_white", 0.0),
                s_ent=s.get({"prop_slot": "s_ent_slot", "prop_answer": "s_ent_answer",
                             "prop_oracle": "s_ent_oracle", "raw": "s_ent_raw"}[cfg.nli_hypothesis], 0.0),
            )
            f = tc.fuse(sig, cfg); v = tc.verdict(f, cfg)
            fused[p["pid"]] = f; verdicts[p["pid"]] = v
            if v in ("CORRECT", "AMBIGUOUS"):
                retained.append(p)
    else:
        retained = list(cands)
        for p in cands:
            fused[p["pid"]] = float(sig_by_pid[p["pid"]]["s_emb"]); verdicts[p["pid"]] = "CORRECT"
    t_ver = time.perf_counter() - t0
    kappa = (len(retained) / len(cands)) if cands else 0.0

    ctx = "\n\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(retained))
    t0 = time.perf_counter()
    if cfg.retriever == "none":
        prompt = tc.PROMPTS["gen_closedbook_v1"].format(question=q)
    else:
        prompt = tc.PROMPTS[cfg.prompt_id].format(context=ctx or "(no context)", question=q)
    pred, n_in = _generate(model_id, prompt)
    t_gen = time.perf_counter() - t0

    t0 = time.perf_counter()
    label = "SKIPPED"
    if cfg.self_check and retained:
        chk, _ = _generate(model_id, tc.PROMPTS["check_v2"].format(
            context=ctx, question=q, answer=pred), max_new_tokens=6)
        label = "PARTIAL" if "PARTIAL" in chk.upper() else \
                "UNSUPPORTED" if "UNSUP" in chk.upper() else "SUPPORTED"
        if label == "PARTIAL":
            pred, _ = _generate(model_id, prompt + "\nUse only the context. Answer:")
    t_sc = time.perf_counter() - t0

    conf = float(np.mean([fused[p["pid"]] for p in retained])) if retained else 0.0
    return {
        "qid": rec["qid"], "config": cfg.name, "generator": model_id,
        "kappa": kappa, "n_cands": len(cands), "n_retained": len(retained),
        "retained_ids": [p["pid"] for p in retained],
        "gold_retained": sum(1 for p in retained if p["is_gold"]),
        "gold_available": sum(1 for p in paras if p["is_gold"]),
        "s_fused": fused, "verdicts": verdicts, "confidence": conf,
        "context_tokens": n_in, "selfcheck": label,
        "pred": pred, "gold": gold,
        "em": tc.exact_match(pred, gold), "f1": tc.token_f1(pred, gold),
        "t_retrieve": t_ret, "t_verify": t_ver, "t_generate": t_gen,
        "t_selfcheck": t_sc, "t_total": t_ret + t_ver + t_gen + t_sc,
        "note": "timings from cloud GPU; NOT the paper's latency numbers",
    }


@app.function(gpu=GEN_GPU, cpu=4, memory=32768, **COMMON)
def ablation_shard(split: str, config_name: str, lo: int, hi: int,
                   model_id: str = GEN_ID, tag: str = "main") -> str:
    _env(); seed_all()
    import pandas as pd
    import triver_core as tc

    out_path = R / "results" / tag / model_id.split("/")[-1] / config_name / f"{lo:05d}.jsonl"
    if out_path.exists():
        return str(out_path)

    frozen = R / "artifacts" / "frozen.json"
    taus = json.loads(frozen.read_text()) if frozen.exists() else {"tau_lo": 0.40, "tau_hi": 0.62}
    cfg = tc.p0_configs(taus["tau_lo"], taus["tau_hi"])[config_name]

    recs = read_jsonl(R / "splits" / f"{split}.jsonl")[lo:hi]
    sig = pd.read_parquet(R / "artifacts" / "signals.parquet").set_index("pid").to_dict("index")

    rows = [_run_one(r, sig, cfg, model_id) for r in recs]
    write_jsonl(out_path, rows)
    results.commit()
    return str(out_path)


@app.function(cpu=8, memory=32768, **COMMON)
def calibrate() -> dict:
    """E5 — sweep thresholds on calib using the cached signals only. Seconds, not hours."""
    import numpy as np, pandas as pd, itertools
    import triver_core as tc

    df = pd.read_parquet(R / "artifacts" / "signals.parquet")
    calib_ids = {r["qid"] for r in read_jsonl(R / "splits" / "calib.jsonl")}
    d = df[df.qid.isin(calib_ids)].copy()

    grid = []
    W = [(1, 0, 0), (0, 0, 1), (.5, 0, .5), (.6, 0, .4), (.4, 0, .6), (.34, .33, .33)]
    for w, tlo, thi in itertools.product(W, np.arange(0.20, 0.75, 0.025),
                                         np.arange(0.30, 0.92, 0.025)):
        if thi <= tlo:
            continue
        w = np.array(w, float); w = w / w.sum()
        s = w[0] * d.s_emb + w[1] * d.s_str_sent_white + w[2] * d.s_ent_slot
        keep = s >= tlo
        kappa = float(keep.mean())
        if kappa >= 0.95 or kappa < 0.15:        # filters nothing / filters everything
            continue
        gold_recall = float(keep[d.is_gold].mean())
        grid.append({"w_emb": w[0], "w_str": w[1], "w_ent": w[2],
                     "tau_lo": float(tlo), "tau_hi": float(thi),
                     "kappa": kappa, "gold_recall": gold_recall,
                     "distractor_reject": float(1 - keep[~d.is_gold].mean())})

    g = pd.DataFrame(grid).sort_values(["gold_recall", "kappa"], ascending=[False, True])
    g.to_csv(R / "results" / "analysis" / "e5_threshold_sweep.csv", index=False)

    best = g[g.gold_recall >= g.gold_recall.max() - 0.02].sort_values("kappa").iloc[0]
    frozen = {"tau_lo": float(best.tau_lo), "tau_hi": float(best.tau_hi),
              "weights": [float(best.w_emb), float(best.w_str), float(best.w_ent)],
              "kappa_calib": float(best.kappa), "gold_recall_calib": float(best.gold_recall),
              "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (R / "artifacts" / "frozen.json").write_text(json.dumps(frozen, indent=2))
    results.commit()

    print(f"\n[FROZEN] {frozen}")
    assert frozen["kappa_calib"] < 0.95, "GATE FAILED: thresholds still not biting"
    with wandb_run("e5_calibrate", frozen):
        import wandb; wandb.log(frozen)
    return frozen


# ============================================================================
# Stage 5 — analysis, tables, figures
# ============================================================================

@app.function(cpu=8, memory=32768, **COMMON)
def analyze(tag: str = "main") -> dict:
    import numpy as np, pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import triver_core as tc

    AN = R / "results" / "analysis"; AN.mkdir(parents=True, exist_ok=True)
    FIG = R / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

    base = R / "results" / tag / GEN_ID.split("/")[-1]
    per_cfg = {}
    for cdir in sorted(base.glob("*")):
        rows = [r for f in sorted(cdir.glob("*.jsonl")) for r in read_jsonl(f)]
        if rows:
            per_cfg[cdir.name] = pd.DataFrame(rows).set_index("qid").sort_index()

    # ---- main table with Wilson intervals
    main = []
    for name, d in per_cfg.items():
        lo, hi = tc.wilson(int(d.em.sum()), len(d))
        main.append({"config": name, "n": len(d),
                     "EM": 100 * d.em.mean(), "CI_lo": 100 * lo, "CI_hi": 100 * hi,
                     "F1": d.f1.mean(), "kappa": d.kappa.mean(),
                     "gold_recall": (d.gold_retained / d.gold_available.clip(lower=1)).mean(),
                     "ctx_tokens": d.context_tokens.mean(),
                     "ECE": tc.ece(d.confidence, d.em)})
    mt = pd.DataFrame(main).sort_values("EM", ascending=False)
    mt.to_csv(AN / "main_table.csv", index=False)

    # ---- paired tests against B2 (the pre-registered comparator)
    tests = []
    if "B2" in per_cfg:
        ref = per_cfg["B2"]
        for name, d in per_cfg.items():
            if name == "B2":
                continue
            common = ref.index.intersection(d.index)
            p, b, c = tc.mcnemar_exact(ref.loc[common].em, d.loc[common].em)
            blo, bhi = tc.paired_bootstrap(ref.loc[common].f1, d.loc[common].f1)
            tests.append({"config": name, "vs": "B2", "n": len(common),
                          "b": b, "c": c, "p_raw": p,
                          "f1_delta": d.loc[common].f1.mean() - ref.loc[common].f1.mean(),
                          "f1_ci_lo": blo, "f1_ci_hi": bhi})
        if tests:
            adj = tc.holm([t["p_raw"] for t in tests])
            for t, a in zip(tests, adj):
                t["p_holm"] = a
    pd.DataFrame(tests).to_csv(AN / "pairwise_tests.csv", index=False)

    # ---- Fig: EM with Wilson intervals
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(mt))
    err = np.vstack([mt.EM - mt.CI_lo, mt.CI_hi - mt.EM])
    ax.bar(x, mt.EM, yerr=err, capsize=4, color="#2980B9", edgecolor="black", linewidth=.6)
    ax.set_xticks(x); ax.set_xticklabels(mt.config); ax.set_ylabel("Exact Match (%)")
    ax.set_title("Accuracy with 95% Wilson intervals"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig_em.pdf"); fig.savefig(FIG / "fig_em.png", dpi=160)

    # ---- Fig: Pareto (context tokens as the hardware-free proxy for cost)
    if len(mt) > 1:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.scatter(mt.ctx_tokens, mt.EM, s=60, color="#27AE60", edgecolor="black")
        for _, r in mt.iterrows():
            ax.annotate(r.config, (r.ctx_tokens, r.EM), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)
        ax.set_xlabel("mean context tokens"); ax.set_ylabel("Exact Match (%)")
        ax.set_title("Accuracy vs context cost"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(FIG / "fig_pareto.pdf")

    # ---- Fig: reliability diagram
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    for name, d in per_cfg.items():
        xs, ys, _ = tc.reliability_bins(d.confidence, d.em)
        ax.plot(xs, ys, marker="o", ms=3, label=name, lw=1.2)
    ax.set_xlabel("mean fused score"); ax.set_ylabel("accuracy")
    ax.set_title("Reliability"); ax.legend(fontsize=6); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig_reliability.pdf")

    # ---- Fig: scale study, if present
    scale_root = R / "results" / "scale"
    if scale_root.exists():
        pts = []
        for mdir in sorted(scale_root.glob("*")):
            for cdir in sorted(mdir.glob("*")):
                rows = [r for f in cdir.glob("*.jsonl") for r in read_jsonl(f)]
                if rows:
                    pts.append({"model": mdir.name, "config": cdir.name,
                                "EM": 100 * np.mean([r["em"] for r in rows])})
        if pts:
            sd = pd.DataFrame(pts)
            sd.to_csv(AN / "e7_scale.csv", index=False)
            piv = sd.pivot(index="model", columns="config", values="EM")
            order = [m.split("/")[-1] for m in SCALE_IDS]
            piv = piv.reindex([m for m in order if m in piv.index])
            fig, ax = plt.subplots(figsize=(5.6, 3.6))
            for c in piv.columns:
                ax.plot(piv.index, piv[c], marker="o", label=c)
            if {"A2", "B2"} <= set(piv.columns):
                ax.plot(piv.index, piv.A2 - piv.B2, marker="s", ls="--",
                        color="black", label="A2 - B2 gap")
            ax.set_ylabel("Exact Match (%)"); ax.set_xlabel("generator")
            ax.set_title("Verification gap vs model scale"); ax.legend(fontsize=7); ax.grid(alpha=.3)
            fig.tight_layout(); fig.savefig(FIG / "fig_scale.pdf")

    # ---- the two experiments that must stay on the edge device
    (R / "results" / "LOCAL_RUNS.md").write_text(f"""# Runs that cannot happen on Modal

E8 (cost model) and E9 (latency) measure the edge device. That is the paper's
premise, so a cloud number cannot substitute.

Frozen config is in `artifacts/frozen.json`. Pull the volume:

    modal volume get triver-results / ./triver-results

Then on the 6.2 GB CPU box, quiesced:

    sudo cpupower frequency-set -g performance
    sudo systemctl stop cron unattended-upgrades snapd
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

    # E9: 60 questions x 8 configs x 5 repeats, CONFIGS INTERLEAVED
    taskset -c 0-3 python local_latency.py \\
        --split timing --repeats 5 --interleave \\
        --frozen artifacts/frozen.json \\
        --out results/timing/

    # E8: context-length sweep for the cost model
    taskset -c 0-3 python local_latency.py --costmodel \\
        --lengths 128,256,384,512,768,1024,1536,2048 \\
        --out results/costmodel/

Push the results back and rerun analyze:

    modal volume put triver-results ./results/timing  /results/timing
    modal volume put triver-results ./results/costmodel /results/costmodel
    modal run modal_research.py --stage analyze

`local_latency.py` must import `_run_one` from this file so the local and cloud
paths are the same code. Report median and IQR, and print sigma_run across the
5 repeats: any latency claim smaller than 2*sigma_run is not a finding.
""")

    results.commit()
    with wandb_run("analyze"):
        import wandb
        wandb.log({"main_table": wandb.Table(dataframe=mt)})
        if tests:
            wandb.log({"pairwise": wandb.Table(dataframe=pd.DataFrame(tests))})
        for f in sorted(FIG.glob("*.png")):
            wandb.log({f.stem: wandb.Image(str(f))})
    print("\n" + mt.to_string(index=False) + "\n")
    return {"configs": list(per_cfg), "table": mt.to_dict("records")}


# ============================================================================
# Orchestration
# ============================================================================

def _shards(n: int, size: int) -> list[tuple[int, int]]:
    return [(i, min(i + size, n)) for i in range(0, n, size)]


@app.local_entrypoint()
def main(stage: str = "gate", shard: int = 25, tag: str = "main"):
    t0 = time.time()

    if stage in ("gate", "all"):
        print(">> prepare"); print(prepare.remote())
        for split, n in (("calib", N_CALIB), ("eval", N_EVAL)):
            jobs = [(split, lo, hi) for lo, hi in _shards(n, shard)]
            print(f">> signals {split}: {len(jobs)} shards")
            list(signals_shard.starmap(jobs))
        print(">> E1/E2/E4 signal AUC")
        auc = signal_auc.remote()
        print(">> E3 NLI model-size probe"); print(nli_probe.remote())
        gate = auc["E2_gate"]
        print(f"\n{'='*64}\n  GATE: {gate}\n{'='*64}\n")
        if gate == "NEGATIVE_RESULT_PAPER" and stage == "all":
            print("Oracle ceiling below 0.70. Stopping before the expensive stages.")
            print("See RESEARCH_VS_TOOL.md Part 5 for the negative-result paper.")
            return

    if stage in ("calib", "all"):
        print(">> E5 calibration"); print(calibrate.remote())

    if stage in ("main", "all"):
        cfgs = ["B0", "B1", "B2", "B4", "B5", "A1", "A2", "S0"]
        jobs = [("eval", c, lo, hi, GEN_ID, tag)
                for c in cfgs for lo, hi in _shards(N_EVAL, shard)]
        print(f">> E6 ablation: {len(jobs)} shards across {len(cfgs)} configs")
        list(ablation_shard.starmap(jobs))

    if stage in ("scale", "all"):
        cfgs = ["B1", "B2", "B4", "A2"]
        jobs = [("eval", c, lo, hi, m, "scale")
                for m in SCALE_IDS for c in cfgs for lo, hi in _shards(300, shard)]
        print(f">> E7 scale study: {len(jobs)} shards")
        list(ablation_shard.starmap(jobs))

    if stage in ("analyze", "gate", "main", "scale", "all"):
        if stage != "gate":
            print(">> analyze"); analyze.remote(tag)

    print(f"\nwall clock {(time.time()-t0)/60:.1f} min")
    print("artifacts: modal volume get triver-results / ./triver-results")
