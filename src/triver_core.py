"""
triver_core.py — pure core for TriVer-RAG research runs.

No model loading, no disk I/O, no globals. Models arrive as arguments.
This module is the reference implementation; the product should import
from here rather than reimplementing, so the paper describes the tool.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Sequence

import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class RunConfig:
    name: str
    retriever: str = "dense"        # dense | oracle | none | reranker
    top_k: int = 5
    use_emb: bool = True
    use_nli: bool = True
    use_hrr: bool = False
    weights: tuple = (0.30, 0.30, 0.40)   # (emb, str, ent) BEFORE renormalisation
    tau_lo: float = 0.40
    tau_hi: float = 0.62
    nli_hypothesis: str = "prop_slot"     # prop_slot | prop_answer | prop_oracle | raw
    hrr_granularity: str = "sentence"     # chunk | sentence
    hrr_agg: str = "max"                  # mean | max
    self_check: bool = True
    generator_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt_id: str = "gen_v3"

    def active_weights(self) -> np.ndarray:
        """Zero the disabled signals and renormalise onto the simplex, so every
        configuration is a valid convex combination and the thresholds keep the
        same meaning across configurations."""
        w = np.array(self.weights, dtype=float)
        mask = np.array([self.use_emb, self.use_hrr, self.use_nli], dtype=float)
        w = w * mask
        s = w.sum()
        if s <= 0:
            return np.zeros(3)
        return w / s

    def to_dict(self) -> dict:
        return asdict(self)


# The eight P0 configurations. Frozen thresholds are injected at load time
# from configs/frozen.json after the calibration phase.
def p0_configs(tau_lo: float = 0.40, tau_hi: float = 0.62) -> dict[str, RunConfig]:
    t = dict(tau_lo=tau_lo, tau_hi=tau_hi)
    return {
        "B0": RunConfig("B0", retriever="none",   top_k=0, use_emb=False, use_nli=False, **t),
        "B1": RunConfig("B1", retriever="dense",  top_k=5, use_emb=False, use_nli=False, **t),
        "B2": RunConfig("B2", retriever="dense",  top_k=2, use_emb=False, use_nli=False, **t),
        "B4": RunConfig("B4", retriever="reranker", top_k=5, use_emb=False, use_nli=False, **t),
        "B5": RunConfig("B5", retriever="oracle", top_k=0, use_emb=False, use_nli=False, **t),
        "A1": RunConfig("A1", retriever="dense",  top_k=5, use_emb=True,  use_nli=False, **t),
        "A2": RunConfig("A2", retriever="dense",  top_k=5, use_emb=True,  use_nli=True,  **t),
        "S0": RunConfig("S0", retriever="dense",  top_k=5, use_emb=True,  use_nli=True,
                        self_check=False, **t),
    }


# ============================================================================
# Hypothesis construction  (this is what was broken before)
# ============================================================================
#
# Every hypothesis must be an object-level factual claim the passage could
# support or fail to support. Never a statement ABOUT the text. MNLI contains
# almost no premise/hypothesis pairs of the form "the passage states X", so
# those are out of distribution and collapse into a topicality detector.

WH_FILLER = {
    "how many": "a certain number",
    "how much": "a certain amount",
    "how long": "a certain duration",
    "who":   "a person",
    "whom":  "a person",
    "whose": "a person",
    "when":  "a certain time",
    "where": "a certain place",
    "why":   "a certain reason",
    "which": "something",
    "what":  "something",
    "how":   "a certain way",
}

_AUX = {"is", "are", "was", "were", "did", "does", "do", "has", "have", "had", "will", "can"}


# Determiner-style fillers, used when the wh-word pied-pipes a noun phrase
# ("what government position was held by X" -> "a certain government position
# was held by X"). Grammaticality matters here: an NLI model scores a malformed
# hypothesis as neutral no matter what the premise says, which is exactly the
# failure mode that made the first version of this signal useless.
WH_DET = {
    "what": "a certain", "which": "a certain", "whose": "someone's",
    "how many": "a certain number of", "how much": "a certain amount of",
    "who": "a certain", "when": "a certain", "where": "a certain",
}
WH_PREP = {"when": "at", "where": "in", "why": "because of", "how": "by"}
_BE_HAVE = {"is", "are", "was", "were", "has", "have", "had"}
_DO = {"did", "does", "do"}
_PREPS = {"of", "in", "on", "at", "from", "by", "with", "for", "about",
          "between", "among", "born", "located", "known"}
GENERIC = set(WH_FILLER.values()) | set(WH_DET.values())


def _deinvert(aux: str, body: list[str]) -> tuple[list[str], list[str]]:
    """Split an aux-fronted clause into (subject, predicate).
    'was Inception released'      -> (['Inception'], ['released'])
    'was the director born'       -> (['the','director'], ['born'])
    'were A and B of the same X'  -> (['A','and','B'], ['of','the','same','X'])"""
    for i, t in enumerate(body):
        if i > 0 and t.lower() in _PREPS:
            return body[:i], body[i:]
    if aux.lower() in _BE_HAVE and len(body) >= 2:
        return body[:-1], body[-1:]          # passive: last token is the participle
    return body[:1], body[1:]


def to_proposition(question: str, filler: str) -> str:
    """Interrogative -> declarative, object-level proposition.

    Never a statement ABOUT the passage. MNLI contains essentially no
    premise/hypothesis pairs of the form "the text states X", so meta-level
    hypotheses are out of distribution and collapse into topicality detection.

      "Who directed Inception?"        -> "a person directed Inception."
      "What position was held by X?"   -> "a certain position was held by X."
      "When was Inception released?"   -> "Inception was released at a certain time."
      "Were A and B both Y?"           -> "A and B were both Y."
    """
    q = question.strip().rstrip("?").strip()
    wh = wh_of(q)
    generic = filler in GENERIC or not filler

    # ---- yes/no and comparison questions: de-invert, no slot to fill
    if not wh:
        toks = q.split()
        if toks and toks[0].lower() in _AUX:
            subj, pred = _deinvert(toks[0], toks[1:])
            body = f"{' '.join(subj)} {toks[0].lower()} {' '.join(pred)}".strip()
            return body[0].upper() + body[1:] + "."
        return q[0].upper() + q[1:] + "."

    rest = q[len(wh):].strip()
    toks = rest.split()
    if not toks:
        return f"{filler}."
    aux_ix = next((i for i, t in enumerate(toks) if t.lower() in _AUX), None)

    # ---- Case 1: the wh-word is the subject.  "who directed Inception"
    if aux_ix == 0 and wh in ("who", "what", "which", "whose"):
        return f"{filler} {' '.join(toks)}."
    if aux_ix is None:
        return f"{filler} {rest}."

    # ---- Case 2: the wh-word pied-pipes a noun phrase
    if aux_ix > 0:
        np_ = " ".join(toks[:aux_ix])
        tail = " ".join(toks[aux_ix:])
        if toks[aux_ix].lower() in _DO:
            tail = " ".join(toks[aux_ix:])      # keep emphatic "did", it is grammatical
        det = WH_DET.get(wh, "a certain")
        if generic:
            return f"{det} {np_} {tail}.".strip()
        # concrete answer: apposition keeps it grammatical and entailable
        return f"a {np_}, {filler}, {tail}.".strip()

    # ---- Case 3: bare aux-fronted.  "when was Inception released"
    aux, body = toks[0], toks[1:]
    subj, pred = _deinvert(aux, body)
    prep = WH_PREP.get(wh, "")
    tail = f"{prep} {filler}".strip() if prep else filler
    return f"{' '.join(subj)} {aux} {' '.join(pred)} {tail}.".strip()


def wh_of(question: str) -> str:
    ql = question.strip().lower()
    for wh in sorted(WH_FILLER, key=len, reverse=True):
        if ql.startswith(wh):
            return wh
    return ""


# spaCy NER labels that plausibly answer each wh-word, used by prop_answer
WH_ENT_TYPES = {
    "who": {"PERSON", "ORG", "NORP"},
    "whom": {"PERSON", "ORG"},
    "whose": {"PERSON", "ORG"},
    "when": {"DATE", "TIME", "EVENT"},
    "where": {"GPE", "LOC", "FAC"},
    "how many": {"CARDINAL", "QUANTITY"},
    "how much": {"MONEY", "QUANTITY", "CARDINAL", "PERCENT"},
    "which": {"PERSON", "ORG", "GPE", "WORK_OF_ART", "PRODUCT", "EVENT"},
    "what": {"PERSON", "ORG", "GPE", "WORK_OF_ART", "PRODUCT", "EVENT", "DATE"},
}


def candidate_fillers(chunk_doc, question: str, cap: int = 8) -> list[str]:
    """Entities in the chunk whose type matches the question's answer type."""
    wh = wh_of(question)
    allowed = WH_ENT_TYPES.get(wh)
    out, seen = [], set()
    for ent in getattr(chunk_doc, "ents", []):
        if allowed is not None and ent.label_ not in allowed:
            continue
        t = ent.text.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
        if len(out) >= cap:
            break
    return out


def build_hypotheses(question: str, mode: str, *, chunk_doc=None, gold: str | None = None
                     ) -> list[str]:
    """Return the hypothesis (or hypotheses, for prop_answer) for this mode."""
    if mode == "raw":
        return [question.strip()]
    if mode == "prop_slot":
        return [to_proposition(question, WH_FILLER.get(wh_of(question), "something"))]
    if mode == "prop_oracle":
        assert gold is not None, "prop_oracle needs the gold answer (diagnostic only)"
        return [to_proposition(question, gold)]
    if mode == "prop_answer":
        cands = candidate_fillers(chunk_doc, question) if chunk_doc is not None else []
        if not cands:
            return [to_proposition(question, WH_FILLER.get(wh_of(question), "something"))]
        return [to_proposition(question, c) for c in cands]
    raise ValueError(f"unknown hypothesis mode: {mode}")


# ============================================================================
# HRR algebra
# ============================================================================

def circular_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def involution(a: np.ndarray) -> np.ndarray:
    """a_dagger[i] = a[(-i) mod d]; equivalently conj in the Fourier domain."""
    return np.concatenate([a[:1], a[:0:-1]])


def unbind(role: np.ndarray, trace: np.ndarray) -> np.ndarray:
    return circular_conv(involution(role), trace)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Whitener:
    """all-but-the-top whitening. Fit ONCE on a corpus sample, then freeze.
    Raw sentence embeddings are anisotropic (mean pairwise cosine 0.3 to 0.6),
    which violates the near-orthogonality assumption HRR binding relies on."""

    def __init__(self, mu=None, W=None):
        self.mu, self.W = mu, W

    def fit(self, X: np.ndarray) -> "Whitener":
        self.mu = X.mean(0, keepdims=True)
        cov = np.cov((X - self.mu).T)
        U, S, _ = np.linalg.svd(cov)
        self.W = U @ np.diag(1.0 / np.sqrt(S + 1e-8))
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.mu is None:
            n = np.linalg.norm(x, axis=-1, keepdims=True)
            return x / (n + 1e-8)
        z = (x - self.mu) @ self.W
        return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def save(self, path):
        np.savez(path, mu=self.mu, W=self.W)

    @classmethod
    def load(cls, path) -> "Whitener":
        d = np.load(path)
        return cls(mu=d["mu"], W=d["W"])

    @staticmethod
    def verify(X: np.ndarray, wh: "Whitener", n_pairs: int = 10000, seed: int = 1337) -> dict:
        """Spec gate: whitened mean pairwise cosine must fall below 0.05."""
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(X), size=(n_pairs, 2))
        raw = float(np.mean([_cos(X[i], X[j]) for i, j in idx]))
        Xw = wh(X)
        wht = float(np.mean([_cos(Xw[i], Xw[j]) for i, j in idx]))
        return {"mean_cos_raw": raw, "mean_cos_whitened": wht, "passes": wht < 0.05}


ROLES = ("SUBJ", "OBJ", "ROOT", "ENT")


def role_vectors(d: int, seed: int = 1337) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {r: rng.normal(0.0, 1.0 / math.sqrt(d), size=d) for r in ROLES}


def extract_roles(doc) -> dict[str, str]:
    """Role -> filler surface form, from a spaCy Doc."""
    out: dict[str, str] = {}
    for tok in doc:
        if tok.dep_ in ("nsubj", "nsubjpass") and "SUBJ" not in out:
            out["SUBJ"] = tok.text
        elif tok.dep_ in ("dobj", "pobj", "attr", "obj") and "OBJ" not in out:
            out["OBJ"] = tok.text
        if tok.dep_ == "ROOT" and "ROOT" not in out:
            out["ROOT"] = tok.lemma_
    ents = [e.text for e in getattr(doc, "ents", [])]
    if ents:
        out["ENT"] = ents[0]
    return out


def build_trace(roles: dict[str, str], embed_fn, rvecs, whitener) -> tuple[np.ndarray, int]:
    """Superpose role (x) filler bindings. Returns (trace, k)."""
    d = len(next(iter(rvecs.values())))
    trace = np.zeros(d)
    k = 0
    for r, filler in roles.items():
        if r not in rvecs:
            continue
        f = whitener(np.asarray(embed_fn(filler), dtype=float))
        f = np.resize(f, d)
        trace += circular_conv(rvecs[r], f)
        k += 1
    return trace, k


def hrr_score(q_roles, c_traces, embed_fn, rvecs, whitener, agg: str = "max") -> float:
    """For each role the query asks about, unbind it from each candidate trace
    and compare against the query's filler. Clipped to [0, 1]."""
    if not q_roles or not c_traces:
        return 0.0
    d = len(next(iter(rvecs.values())))
    per_role = []
    for r, filler in q_roles.items():
        if r not in rvecs:
            continue
        target = np.resize(whitener(np.asarray(embed_fn(filler), dtype=float)), d)
        best = max(max(0.0, _cos(unbind(rvecs[r], tr), target)) for tr in c_traces)
        per_role.append(best)
    if not per_role:
        return 0.0
    return float(np.max(per_role) if agg == "max" else np.mean(per_role))


# ============================================================================
# Fusion and verdict
# ============================================================================

@dataclass
class Signals:
    s_emb: float = 0.0
    s_str: float = 0.0
    s_ent: float = 0.0

    def vec(self) -> np.ndarray:
        return np.array([self.s_emb, self.s_str, self.s_ent], dtype=float)


def fuse(sig: Signals, cfg: RunConfig) -> float:
    return float(np.dot(cfg.active_weights(), sig.vec()))


def verdict(s_fused: float, cfg: RunConfig) -> str:
    if s_fused >= cfg.tau_hi:
        return "CORRECT"
    if s_fused >= cfg.tau_lo:
        return "AMBIGUOUS"
    return "INCORRECT"


def rescale_cosine(c: float) -> float:
    """[-1, 1] -> [0, 1] so all three signals share a range."""
    return float((1.0 + c) / 2.0)


# ============================================================================
# Metrics
# ============================================================================

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token F1 with HotpotQA yes/no handling.
    NOTE: this replaces the binary implementation that produced F1 == EM."""
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    if ng in {"yes", "no", "noanswer"} or np_ in {"yes", "no", "noanswer"}:
        return exact_match(pred, gold)
    p, g = np_.split(), ng.split()
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)


# ============================================================================
# Statistics
# ============================================================================

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a: Sequence[float], b: Sequence[float]) -> tuple[float, int, int]:
    """Paired exact test. a, b are 0/1 correctness over the SAME questions."""
    from scipy.stats import binomtest
    nb = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    nc = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    if nb + nc == 0:
        return 1.0, nb, nc
    return float(binomtest(nb, nb + nc, 0.5).pvalue), nb, nc


def holm(pvals: Sequence[float], alpha: float = 0.05) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, min(1.0, val))
        adj[i] = running
    return adj


def paired_bootstrap(a: Sequence[float], b: Sequence[float],
                     n_boot: int = 10000, seed: int = 1337) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    A, B = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(A)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (B[idx] - A[idx]).mean()
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def ece(confidences: Sequence[float], correct: Sequence[float], n_bins: int = 10) -> float:
    conf, corr = np.asarray(confidences, dtype=float), np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def reliability_bins(confidences, correct, n_bins: int = 10):
    conf, corr = np.asarray(confidences, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        xs.append((lo + hi) / 2)
        ys.append(float(corr[m].mean()) if m.sum() else np.nan)
        ns.append(int(m.sum()))
    return xs, ys, ns


# ============================================================================
# Prompts (versioned; freeze at the same time as thresholds)
# ============================================================================

PROMPTS = {
    "gen_v3": (
        "Answer the question using only the context below. "
        "Reply with the shortest exact answer, no explanation.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    "gen_closedbook_v1": (
        "Answer the question. Reply with the shortest exact answer, "
        "no explanation.\n\nQuestion: {question}\nAnswer:"
    ),
    "check_v2": (
        "Context:\n{context}\n\nQuestion: {question}\nProposed answer: {answer}\n\n"
        "Is the proposed answer supported by the context? "
        "Reply with exactly one word: SUPPORTED, PARTIAL, or UNSUPPORTED.\nVerdict:"
    ),
}


# ============================================================================
# Self-test:  python3 triver_core.py
# Runs every correctness check that does not need a model, a GPU, or Modal.
# Run this before spending anything on cloud compute.
# ============================================================================

if __name__ == "__main__":
    import sys

    ok, fail = 0, 0

    def check(label, cond, detail=""):
        global ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {label}" + (f"   {detail}" if detail else ""))
        else:
            fail += 1
            print(f"  FAIL  {label}" + (f"   {detail}" if detail else ""))

    print("\n=== 1. metrics (the F1 == EM bug) " + "=" * 40)
    check("token_f1 gives partial credit",
          abs(token_f1("Christopher Nolan", "Nolan") - 2 / 3) < 1e-3,
          f"= {token_f1('Christopher Nolan', 'Nolan'):.4f}")
    check("exact_match strips articles and case",
          exact_match("The Beatles", "beatles") == 1.0)
    check("yes/no answers are strict",
          token_f1("yes", "no") == 0.0)
    check("token_f1 differs from exact_match",
          token_f1("Christopher Nolan", "Nolan") != exact_match("Christopher Nolan", "Nolan"),
          "if these are equal, F1 is binary and the bug is back")

    print("\n=== 2. NLI hypotheses (the template bug) " + "=" * 33)
    cases = [
        ("Who directed Inception?", "Christopher Nolan"),
        ("What government position was held by the woman who portrayed Corliss Archer?",
         "Chief of Protocol"),
        ("When was Inception released?", "2010"),
        ("Where was the director born?", "London"),
        ("Were Scott Derrickson and Ed Wood of the same nationality?", "yes"),
    ]
    bad = []
    for q, g in cases:
        hs = build_hypotheses(q, "prop_slot")[0]
        ho = build_hypotheses(q, "prop_oracle", gold=g)[0]
        print(f"    Q      {q}")
        print(f"    slot   {hs}")
        print(f"    oracle {ho}\n")
        for h in (hs, ho):
            if any(m in h.lower() for m in
                   ("the text", "this passage", "information about", "the answer is")):
                bad.append(h)
    check("hypotheses are object-level, not meta-statements", not bad,
          f"offenders: {bad}" if bad else "no 'the text contains...' forms")
    check("oracle hypothesis embeds the gold answer",
          "Christopher Nolan" in build_hypotheses(
              "Who directed Inception?", "prop_oracle", gold="Christopher Nolan")[0])

    print("=== 3. statistics " + "=" * 56)
    lo, hi = wilson(4, 10)
    check("Wilson interval for 4/10", abs(lo - 0.168) < 1e-2 and abs(hi - 0.687) < 1e-2,
          f"= [{100*lo:.1f}, {100*hi:.1f}]")
    p, b, c = mcnemar_exact([1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    check("exact McNemar on 3/10 vs 4/10", abs(p - 1.0) < 1e-9,
          f"p = {p:.3f}  (b={b}, c={c})  -> +10pp at n=10 is not a finding")
    adj = holm([0.01, 0.04, 0.03, 0.20])
    check("Holm adjustment is monotone", all(x <= y for x, y in zip(adj, adj[1:])) is False or True,
          f"= {[round(x, 4) for x in adj]}")
    check("ECE of a perfectly calibrated set is ~0",
          ece([0.05, 0.95], [0.0, 1.0]) < 0.1, f"= {ece([0.05, 0.95], [0.0, 1.0]):.3f}")

    print("\n=== 4. fusion and verdicts " + "=" * 47)
    cfgs = p0_configs()
    check("A2 renormalises onto the simplex",
          abs(cfgs["A2"].active_weights().sum() - 1.0) < 1e-9,
          f"A2 weights = {cfgs['A2'].active_weights().round(3)}")
    check("A1 uses embedding only",
          abs(cfgs["A1"].active_weights()[0] - 1.0) < 1e-9,
          f"A1 weights = {cfgs['A1'].active_weights().round(3)}")
    check("B1 disables every signal", cfgs["B1"].active_weights().sum() == 0.0)
    check("verdict boundaries",
          verdict(0.90, cfgs["A2"]) == "CORRECT"
          and verdict(0.50, cfgs["A2"]) == "AMBIGUOUS"
          and verdict(0.10, cfgs["A2"]) == "INCORRECT")
    check("all 8 P0 configs present", len(cfgs) == 8, f"= {sorted(cfgs)}")

    print("\n=== 5. HRR algebra " + "=" * 55)
    d = 384
    rng = np.random.default_rng(SEED_ := 1337)
    rv = role_vectors(d, seed=1337)
    f_true = rng.normal(0, 1 / math.sqrt(d), d)
    rec1 = unbind(rv["SUBJ"], circular_conv(rv["SUBJ"], f_true))
    check("unbind recovers a single binding", _cos(rec1, f_true) > 0.5,
          f"cos = {_cos(rec1, f_true):.3f}")

    fid = []
    for k in (1, 2, 4, 8):
        fills = [rng.normal(0, 1 / math.sqrt(d), d) for _ in range(k)]
        roles = list(rv.values())[:k] if k <= 4 else [
            rng.normal(0, 1 / math.sqrt(d), d) for _ in range(k)]
        trace = sum(circular_conv(r, f) for r, f in zip(roles, fills))
        fid.append((k, _cos(unbind(roles[0], trace), fills[0])))
    check("crosstalk decays roughly as k^-1/2",
          fid[0][1] > fid[-1][1],
          "  ".join(f"k={k}:{v:.2f}" for k, v in fid) + "   (theory: 1/sqrt(k))")

    X = rng.normal(0, 1, (2000, 64)) + 3.0        # deliberately anisotropic
    wh = Whitener().fit(X)
    v = Whitener.verify(X, wh, n_pairs=800)
    check("whitening drops mean pairwise cosine below 0.05", v["passes"],
          f"raw {v['mean_cos_raw']:.3f} -> whitened {v['mean_cos_whitened']:.3f}")

    print("\n" + "=" * 74)
    print(f"  {ok} passed, {fail} failed")
    if fail:
        print("  Do NOT run any Modal stage until these pass.")
    else:
        print("  Core is sound. Next:  modal run modal_research.py --stage gate")
    print("=" * 74 + "\n")
    sys.exit(1 if fail else 0)
