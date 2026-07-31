#!/usr/bin/env python3
"""
test_core.py — standalone checks for triver_core.py

Works with any version of triver_core.py, including one without a __main__
block. Put this file next to triver_core.py and run:

    python3 test_core.py
"""
import sys
import os

print("=" * 74)
print("  environment")
print("=" * 74)
print(f"  python      {sys.version.split()[0]}")
print(f"  cwd         {os.getcwd()}")
print(f"  files here  {sorted(f for f in os.listdir('.') if f.endswith('.py'))}")

if not os.path.exists("triver_core.py"):
    print("\n  ERROR: triver_core.py is not in this directory.")
    print("  cd into the folder that contains it, or copy it here.")
    sys.exit(1)

for mod in ("numpy", "scipy"):
    try:
        __import__(mod)
        print(f"  {mod:<11} OK")
    except ImportError:
        print(f"  {mod:<11} MISSING  ->  pip install numpy scipy scikit-learn")
        sys.exit(1)

sys.path.insert(0, os.getcwd())
try:
    import triver_core as tc
except Exception as e:
    print(f"\n  ERROR importing triver_core: {type(e).__name__}: {e}")
    sys.exit(1)

has_main = "__main__" in open("triver_core.py").read()
print(f"  triver_core imported OK  (has self-test block: {has_main})")
if not has_main:
    print("  NOTE: your triver_core.py is the older copy without the self-test.")
    print("        This script tests it anyway.")

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}" + (f"   {detail}" if detail else ""))
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"   {detail}" if detail else ""))


print("\n" + "=" * 74)
print("  1. metrics  (the F1 == EM bug)")
print("=" * 74)
f1 = tc.token_f1("Christopher Nolan", "Nolan")
check("token_f1 gives partial credit", abs(f1 - 2 / 3) < 1e-3, f"= {f1:.4f}")
check("exact_match strips articles and case", tc.exact_match("The Beatles", "beatles") == 1.0)
check("yes/no is strict", tc.token_f1("yes", "no") == 0.0)
check("F1 is not binary", f1 != tc.exact_match("Christopher Nolan", "Nolan"))

print("\n" + "=" * 74)
print("  2. NLI hypotheses  (the template bug)")
print("=" * 74)
cases = [
    ("Who directed Inception?", "Christopher Nolan"),
    ("When was Inception released?", "2010"),
    ("Where was the director born?", "London"),
]
bad = []
for q, g in cases:
    hs = tc.build_hypotheses(q, "prop_slot")[0]
    ho = tc.build_hypotheses(q, "prop_oracle", gold=g)[0]
    print(f"    Q      {q}")
    print(f"    slot   {hs}")
    print(f"    oracle {ho}\n")
    for h in (hs, ho):
        if any(m in h.lower() for m in
               ("the text", "this passage", "information about", "the answer is")):
            bad.append(h)
check("hypotheses are object-level, not statements about the text", not bad,
      f"offenders: {bad}" if bad else "clean")

print("=" * 74)
print("  3. statistics")
print("=" * 74)
lo, hi = tc.wilson(4, 10)
check("Wilson interval for 4/10", abs(lo - 0.168) < 1e-2 and abs(hi - 0.687) < 1e-2,
      f"= [{100 * lo:.1f}, {100 * hi:.1f}]")
p, b, c = tc.mcnemar_exact([1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
check("exact McNemar, 3/10 vs 4/10", abs(p - 1.0) < 1e-9,
      f"p = {p:.3f}  (b={b}, c={c})")

print("\n" + "=" * 74)
print("  4. fusion and verdicts")
print("=" * 74)
cfgs = tc.p0_configs()
check("all 8 P0 configs present", len(cfgs) == 8, f"= {sorted(cfgs)}")
check("A2 renormalises onto the simplex",
      abs(cfgs["A2"].active_weights().sum() - 1.0) < 1e-9,
      f"A2 = {cfgs['A2'].active_weights().round(3)}")
check("verdict boundaries",
      tc.verdict(0.90, cfgs["A2"]) == "CORRECT"
      and tc.verdict(0.50, cfgs["A2"]) == "AMBIGUOUS"
      and tc.verdict(0.10, cfgs["A2"]) == "INCORRECT")

print("\n" + "=" * 74)
print("  5. HRR algebra")
print("=" * 74)
import math
import numpy as np

d = 384
rng = np.random.default_rng(1337)
rv = tc.role_vectors(d, seed=1337)
f_true = rng.normal(0, 1 / math.sqrt(d), d)
rec = tc.unbind(rv["SUBJ"], tc.circular_conv(rv["SUBJ"], f_true))
check("unbind recovers a single binding", tc._cos(rec, f_true) > 0.5,
      f"cos = {tc._cos(rec, f_true):.3f}")

fid = []
for k in (1, 2, 4, 8):
    fills = [rng.normal(0, 1 / math.sqrt(d), d) for _ in range(k)]
    roles = [rng.normal(0, 1 / math.sqrt(d), d) for _ in range(k)]
    trace = sum(tc.circular_conv(r, f) for r, f in zip(roles, fills))
    fid.append((k, tc._cos(tc.unbind(roles[0], trace), fills[0])))
check("crosstalk decays with k", fid[0][1] > fid[-1][1],
      "  ".join(f"k={k}:{v:.2f}" for k, v in fid) + "   (theory: 1/sqrt(k))")

X = rng.normal(0, 1, (2000, 64)) + 3.0
wh = tc.Whitener().fit(X)
v = tc.Whitener.verify(X, wh, n_pairs=800)
check("whitening drops mean pairwise cosine below 0.05", v["passes"],
      f"raw {v['mean_cos_raw']:.3f} -> whitened {v['mean_cos_whitened']:.3f}")

print("\n" + "=" * 74)
print(f"  {ok} passed, {fail} failed")
print("  " + ("Core is sound.  Next:  modal run modal_research.py --stage gate"
               if not fail else "Do NOT run any Modal stage until these pass."))
print("=" * 74 + "\n")
sys.exit(1 if fail else 0)
