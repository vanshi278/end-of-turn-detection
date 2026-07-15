"""THE key experiment: train against the cost structure, not against accuracy.

A hold pause only causes a false cutoff if its duration exceeds the agent's
action delay. At a 0.65 s delay, 90% of Hindi holds are already harmless -- the
user resumes first. Standard training spends most of its capacity separating eot
from those harmless short holds, which the metric does not score.

So: weight each hold by how much it can actually cost us. `dur` is used HERE, at
training time, from the labels -- exactly like class_weight. It never becomes a
feature and is never available at inference. See eotlib/features.py.
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/vanshikaagarwal/speedrun")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from eotlib.data import build_many
from eotlib.cv import evaluate, report

X, df = build_many(["eot_handout/eot_data/english", "eot_handout/eot_data/hindi"])
y, dur = df.y.values, df.dur.values


def lr(C=0.1):
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))])


def rf():
    return RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=8,
                                  class_weight="balanced", random_state=0, n_jobs=-1)


def cost_weights(dur, y, d_ref, w_min=0.05, soft=0.0):
    """Weight a hold by its chance of actually costing us a false cutoff.

    hard  (soft=0): 1.0 if it outlives the agent's delay, w_min otherwise
    soft  (soft>0): logistic ramp around d_ref -- hedges against not knowing
                    which delay the scorer will land on
    """
    w = np.ones(len(y))
    hold = y == 0
    if soft > 0:
        w[hold] = w_min + (1 - w_min) / (1 + np.exp(-(dur[hold] - d_ref) / soft))
    else:
        w[hold] = np.where(dur[hold] > d_ref, 1.0, w_min)
    return w


print("=== BASELINE: unweighted (every hold counts equally) ===")
report("logreg C=0.1", evaluate(lr(0.1), X, df))
report("rf depth5", evaluate(rf(), X, df))

print("\n=== HARD cost weighting: only holds that outlive the delay matter ===")
for d_ref in [0.35, 0.45, 0.55, 0.65, 0.75]:
    for w_min in [0.0, 0.05, 0.2]:
        w = cost_weights(dur, y, d_ref, w_min)
        report(f"logreg d_ref={d_ref} w_min={w_min}", evaluate(lr(0.1), X, df, sample_weight=w))

print("\n=== SOFT cost weighting (logistic ramp; robust to the unknown delay) ===")
for d_ref in [0.45, 0.55, 0.65]:
    for s in [0.08, 0.15, 0.30]:
        w = cost_weights(dur, y, d_ref, 0.05, soft=s)
        report(f"logreg d_ref={d_ref} soft={s}", evaluate(lr(0.1), X, df, sample_weight=w))

print("\n=== SOFT weighting + random forest ===")
for d_ref in [0.45, 0.55, 0.65]:
    w = cost_weights(dur, y, d_ref, 0.05, soft=0.15)
    report(f"rf d_ref={d_ref} soft=0.15", evaluate(rf(), X, df, sample_weight=w))
