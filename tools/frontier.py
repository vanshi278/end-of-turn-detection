"""Why is Hindi pinned at exactly 850 ms for every model we try?

Works out what the metric actually demands, rather than guessing. For each
candidate action delay d we ask: how many turns even CAN be interrupted at d
(i.e. contain a hold longer than d), and what eot recall would we need at d to
beat the silence baseline?
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/Users/vanshikaagarwal/speedrun")
from eotlib.data import build_many
from eotlib.metric import auc_score

X, df = build_many(["eot_handout/eot_data/english", "eot_handout/eot_data/hindi"])

for corpus, base in [("english", 1.600), ("hindi", 0.850)]:
    d_ = df[df.corpus == corpus]
    print("=" * 78)
    print(f"{corpus.upper()}   silence baseline = {base*1000:.0f} ms")
    n_turns = d_.group.nunique()
    print(f"\n  {'delay':>6s} {'turns w/ hold>d':>16s} {'%':>6s} {'recall needed':>14s} {'feasible?':>10s}")
    for d in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0, 1.2]:
        risky = d_[(d_.y == 0) & (d_.dur > d)]
        n_risky_turns = risky.group.nunique()
        # latency = d*R + 1.6*(1-R) < base  =>  R > (1.6-base)/(1.6-d)
        need = (1.6 - base) / (1.6 - d)
        feas = "yes" if need <= 1.0 else "IMPOSSIBLE"
        print(f"  {d:6.2f} {n_risky_turns:16d} {100*n_risky_turns/n_turns:5.1f}% "
              f"{need:13.1%} {feas:>10s}")

    print(f"\n  -> at any delay d, at most 5% of turns ({int(0.05*n_turns)} turns) may fire on a hold>d")

    # The AUC that actually matters: eot vs the holds that COST something.
    print(f"\n  METRIC-RELEVANT SEPARABILITY (eot vs holds longer than d):")
    print(f"  {'delay':>6s} {'n_long_holds':>13s} {'best single feature':>22s} {'AUC':>6s}")
    from eotlib.features import FEATURE_NAMES
    for d in [0.35, 0.55, 0.75, 0.95]:
        m = ((d_.y == 1) | ((d_.y == 0) & (d_.dur > d))).values
        sub = d_[m]
        Xi = X[df.corpus.values == corpus][m]
        if sub.y.nunique() < 2 or (sub.y == 0).sum() < 5:
            print(f"  {d:6.2f} {int((sub.y==0).sum()):13d}   (too few long holds)")
            continue
        aucs = [(abs(auc_score(Xi[:, i], sub.y.values) - .5) + .5, FEATURE_NAMES[i])
                for i in range(len(FEATURE_NAMES))]
        best_a, best_n = max(aucs)
        print(f"  {d:6.2f} {int((sub.y==0).sum()):13d} {best_n:>22s} {best_a:6.3f}")

print("=" * 78)
print("\nHOLD DURATION DISTRIBUTION (what fraction of holds are FREE at each delay)")
for corpus in ["english", "hindi"]:
    h = df[(df.corpus == corpus) & (df.y == 0)].dur
    print(f"\n  {corpus}: n={len(h)}")
    for d in [0.25, 0.45, 0.65, 0.85, 1.05]:
        print(f"    delay {d:.2f}s -> {100*(h<=d).mean():5.1f}% of holds are free "
              f"(user resumes before the agent fires)")
