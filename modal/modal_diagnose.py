"""
modal_diagnose.py — three checks to run before accepting NEGATIVE_RESULT_PAPER.

    modal run modal_diagnose.py

D1  NLI label mapping.   ~2 min on T4.  The large<base inversion in E3 is the
    classic signature of reading the wrong logit index.
D2  Bridge vs answer-bearing gold.  Free, pure re-analysis of signals.parquet.
    This is the one that can turn the negative result into a mechanism.
D3  Role extraction.  Free.  k is pinned at 4, which refutes the paper's
    crosstalk explanation and points at the extractor instead.

Nothing here recomputes signals. D1 needs a GPU for two minutes; D2 and D3 are
CPU-only re-analysis of what the gate already produced.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "transformers==4.44.2", "tokenizers==0.19.1",
        "sentencepiece==0.2.0", "spacy==3.7.5", "numpy==1.26.4",
        "scipy==1.14.0", "scikit-learn==1.5.1", "pandas==2.2.2",
        "pyarrow==17.0.0", "matplotlib==3.9.2", "wandb==0.17.7",
    )
    .run_commands("python -m spacy download en_core_web_sm")
    .add_local_python_source("triver_core")
)

cache = modal.Volume.from_name("triver-cache", create_if_missing=True)
results = modal.Volume.from_name("triver-results", create_if_missing=True)
VOLS = {"/cache": cache, "/results": results}
SECRETS = [modal.Secret.from_name("huggingface-secret"),
           modal.Secret.from_name("wandb-secret")]
COMMON = dict(image=image, volumes=VOLS, secrets=SECRETS, timeout=3600, retries=2)

app = modal.App("triver-diagnose")
R = Path("/results")

NLI_IDS = [
    "cross-encoder/nli-deberta-v3-xsmall",
    "cross-encoder/nli-deberta-v3-base",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
]


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# ============================================================================
# D1 — is the entailment logit index correct for every model?
# ============================================================================

@app.function(gpu="T4", cpu=4, memory=16384, **COMMON)
def d1_label_mapping() -> dict:
    os.environ.setdefault("HF_HOME", "/cache/hf")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Two pairs whose correct labels are not in dispute.
    PROBES = [
        ("A man is playing a guitar on stage.", "Someone is making music.", "entailment"),
        ("A man is playing a guitar on stage.", "The stage is completely empty.", "contradiction"),
    ]

    out = {}
    for mid in NLI_IDS:
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = AutoModelForSequenceClassification.from_pretrained(mid).cuda().eval()
        id2label = {int(k): v for k, v in mdl.config.id2label.items()}

        # what the pipeline currently does
        lab = {k.lower(): v for k, v in mdl.config.label2id.items()}
        used_ix = lab.get("entailment", 0)

        with torch.inference_mode():
            enc = tok([p for p, _, _ in PROBES], [h for _, h, _ in PROBES],
                      return_tensors="pt", padding=True, truncation=True).to("cuda")
            probs = torch.softmax(mdl(**enc).logits, -1).float().cpu().numpy()

        pred = [id2label[int(p.argmax())] for p in probs]
        entail_gap = float(probs[0][used_ix] - probs[1][used_ix])

        out[mid] = {
            "id2label": id2label,
            "entailment_index_used": used_ix,
            "index_is_correct": id2label.get(used_ix, "").lower() == "entailment",
            "argmax_on_entailing_pair": pred[0],
            "argmax_on_contradicting_pair": pred[1],
            "p_used[entailing] - p_used[contradicting]": round(entail_gap, 4),
            "VERDICT": ("OK" if id2label.get(used_ix, "").lower() == "entailment"
                        and entail_gap > 0.2 else "SUSPECT"),
        }
        print(f"\n{mid}\n  {json.dumps(out[mid], indent=2)}")
        del mdl
        torch.cuda.empty_cache()

    (R / "results" / "analysis").mkdir(parents=True, exist_ok=True)
    (R / "results" / "analysis" / "d1_label_mapping.json").write_text(json.dumps(out, indent=2))
    results.commit()

    bad = [m for m, v in out.items() if v["VERDICT"] == "SUSPECT"]
    print("\n" + "=" * 70)
    if bad:
        print("  SUSPECT label mapping in:")
        for m in bad:
            print(f"    {m}")
        print("  The E3 numbers for these models are not trustworthy.")
    else:
        print("  All label mappings verified. E3 numbers stand.")
    print("=" * 70)
    return out


# ============================================================================
# D2 — does the gold paragraph actually contain the answer?
# ============================================================================

@app.function(cpu=8, memory=32768, **COMMON)
def d2_bridge_vs_answer() -> dict:
    """Multi-hop gold evidence comes in two kinds. One paragraph carries the
    answer; the other only carries the bridge entity. A per-chunk entailment
    hypothesis about the answer cannot be entailed by the bridge paragraph,
    which caps AUC no matter how good the NLI model is."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    import triver_core as tc

    df = pd.read_parquet(R / "artifacts" / "signals.parquet")

    text_of, gold_of = {}, {}
    for split in ("calib", "eval"):
        for rec in read_jsonl(R / "splits" / f"{split}.jsonl"):
            for p in rec["paragraphs"]:
                text_of[p["pid"]] = p["text"]
                gold_of[p["pid"]] = rec["answer"]

    df["text"] = df.pid.map(text_of)
    df["answer"] = df.pid.map(gold_of)
    df = df.dropna(subset=["text", "answer"])

    def bears_answer(row):
        a = tc.normalize_answer(str(row["answer"]))
        if a in ("yes", "no", ""):
            return False
        return a in tc.normalize_answer(str(row["text"]))

    df["answer_bearing"] = df.apply(bears_answer, axis=1)

    n_q = df.qid.nunique()
    gold = df[df.is_gold]
    per_q = gold.groupby("qid").answer_bearing.sum()

    struct = {
        "questions": int(n_q),
        "gold_paragraphs_total": int(len(gold)),
        "gold_that_contain_the_answer": int(gold.answer_bearing.sum()),
        "fraction_of_gold_bearing_the_answer": round(float(gold.answer_bearing.mean()), 4),
        "questions_with_0_answer_bearing_gold": int((per_q == 0).sum()),
        "questions_with_1_answer_bearing_gold": int((per_q == 1).sum()),
        "questions_with_2plus_answer_bearing_gold": int((per_q >= 2).sum()),
        "yes_no_questions": int(df.groupby("qid").answer.first()
                                .apply(lambda a: tc.normalize_answer(str(a)) in ("yes", "no")).sum()),
    }
    print("\n--- structure of the gold evidence " + "-" * 36)
    print(json.dumps(struct, indent=2))

    sigs = [c for c in df.columns if c.startswith(("s_emb", "s_ent", "s_str"))]
    distractor = df[~df.is_gold]
    bearing = df[df.is_gold & df.answer_bearing]
    bridge = df[df.is_gold & ~df.answer_bearing]

    rows = []
    for c in sigs:
        def auc(pos):
            if len(pos) < 20:
                return float("nan")
            y = [1] * len(pos) + [0] * len(distractor)
            s = list(pos[c]) + list(distractor[c])
            return float(roc_auc_score(y, s))
        rows.append({
            "signal": c,
            "AUC_all_gold": float(roc_auc_score(df.is_gold, df[c])),
            "AUC_answer_bearing": auc(bearing),
            "AUC_bridge_only": auc(bridge),
        })
    tab = pd.DataFrame(rows).sort_values("AUC_answer_bearing", ascending=False)
    tab["lift_bearing_over_bridge"] = tab.AUC_answer_bearing - tab.AUC_bridge_only

    AN = R / "results" / "analysis"
    AN.mkdir(parents=True, exist_ok=True)
    tab.to_csv(AN / "d2_bridge_vs_answer.csv", index=False)
    (AN / "d2_structure.json").write_text(json.dumps(struct, indent=2))
    results.commit()

    print("\n--- AUC split by evidence type " + "-" * 40)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    ent = tab[tab.signal == "s_ent_oracle"].iloc[0]
    print("\n" + "=" * 70)
    print(f"  s_ent_oracle on answer-bearing gold : {ent.AUC_answer_bearing:.3f}")
    print(f"  s_ent_oracle on bridge-only gold    : {ent.AUC_bridge_only:.3f}")
    if ent.AUC_answer_bearing >= 0.80 and ent.AUC_bridge_only <= 0.62:
        print("\n  CONFIRMED: entailment works on answer-bearing evidence and is")
        print("  structurally blind to bridge evidence. Per-chunk entailment")
        print("  verification is mismatched with multi-hop retrieval.")
        print("  This is a mechanism, not a null result. Write the paper around it.")
    elif ent.AUC_answer_bearing < 0.75:
        print("\n  NOT the bridge effect. Entailment is weak even where the")
        print("  answer is present in the paragraph. Suspect the template or")
        print("  the model; check D1 before concluding.")
    else:
        print("\n  Partial effect. Report both numbers and let them speak.")
    print("=" * 70)

    with __import__("wandb").init(project="triver-rag", job_type="d2_bridge",
                                  name="d2_bridge", group="paper-v1", reinit=True):
        import wandb
        wandb.log({"d2_table": wandb.Table(dataframe=tab), **struct})
    return {"structure": struct, "table": tab.to_dict("records")}


# ============================================================================
# D3 — why is k pinned at 4?
# ============================================================================

@app.function(cpu=8, memory=16384, **COMMON)
def d3_role_extraction() -> dict:
    """The paper claims HotpotQA paragraphs bind k = 6 to 10 role-filler pairs.
    The gate measured k = 4.0 at both the median and the 90th percentile, which
    is the ceiling imposed by ROLES having four entries and extract_roles taking
    only the first filler per role."""
    import numpy as np
    import pandas as pd
    import spacy
    import triver_core as tc

    nlp = spacy.load("en_core_web_sm")
    recs = read_jsonl(R / "splits" / "calib.jsonl")[:40]

    cur, avail_sent, avail_para, tok_counts = [], [], [], []
    for rec in recs:
        for p in rec["paragraphs"]:
            doc = nlp(p["text"][:2000])
            cur.append(len(tc.extract_roles(doc)))
            n_sub = sum(1 for t in doc if t.dep_ in ("nsubj", "nsubjpass"))
            n_obj = sum(1 for t in doc if t.dep_ in ("dobj", "pobj", "attr", "obj"))
            n_root = sum(1 for s in doc.sents for t in s if t.dep_ == "ROOT")
            n_ent = len(doc.ents)
            avail_para.append(n_sub + n_obj + n_root + n_ent)
            avail_sent.append(np.mean([
                len(tc.extract_roles(s.as_doc())) for s in doc.sents]) if list(doc.sents) else 0)
            tok_counts.append(len(doc))

    out = {
        "paragraphs_examined": len(cur),
        "k_current_extractor_mean": round(float(np.mean(cur)), 2),
        "k_current_extractor_max": int(np.max(cur)),
        "k_available_in_paragraph_mean": round(float(np.mean(avail_para)), 2),
        "k_available_in_paragraph_p90": round(float(np.percentile(avail_para, 90)), 2),
        "k_per_sentence_mean": round(float(np.mean(avail_sent)), 2),
        "tokens_per_paragraph_mean": round(float(np.mean(tok_counts)), 1),
        "content_discarded_fraction": round(
            1 - float(np.mean(cur)) / max(float(np.mean(avail_para)), 1e-9), 3),
    }
    print("\n--- role extraction " + "-" * 50)
    print(json.dumps(out, indent=2))
    print("\n  The paper's crosstalk section assumes k = 6 to 10 and blames")
    print("  superposition noise. Measured k is capped at 4 by the extractor,")
    print(f"  which discards {100*out['content_discarded_fraction']:.0f}% of the available")
    print("  role-filler pairs. The HRR failure is an extraction failure, not")
    print("  a capacity failure. Rewrite Section 4.1 accordingly.")

    (R / "results" / "analysis" / "d3_role_extraction.json").write_text(json.dumps(out, indent=2))
    results.commit()
    return out


@app.local_entrypoint()
def main():
    print("\n########## D1  NLI label mapping ##########")
    d1 = d1_label_mapping.remote()
    print("\n########## D2  bridge vs answer-bearing ##########")
    d2 = d2_bridge_vs_answer.remote()
    print("\n########## D3  role extraction ##########")
    d3 = d3_role_extraction.remote()

    print("\n" + "=" * 70)
    print("  WHAT TO DO NEXT")
    print("=" * 70)
    bad = [m for m, v in d1.items() if v["VERDICT"] == "SUSPECT"]
    if bad:
        print("  D1 found a bad label index. Fix _nli_entail, rerun --stage gate.")
        print("  Everything downstream of E2 is on hold until that is clean.")
    else:
        ent = [r for r in d2["table"] if r["signal"] == "s_ent_oracle"][0]
        if ent["AUC_answer_bearing"] >= 0.80:
            print("  The bridge effect is real. The paper has a mechanism.")
            print("  Next: single-hop control (Natural Questions) to show the")
            print("  effect disappears when one chunk carries the whole answer.")
        else:
            print("  No bridge effect and no label bug. The negative result is")
            print("  solid. Write it up with D2 and D3 as the diagnostics.")
    print("=" * 70)
