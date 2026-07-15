"""Baseline model sweep. Everything is scored on the REAL metric, OOF by turn."""
import sys
import numpy as np
sys.path.insert(0, "/Users/vanshikaagarwal/speedrun")

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from eotlib.data import build_many
from eotlib.cv import evaluate, cross_lingual, report

D = ["eot_handout/eot_data/english", "eot_handout/eot_data/hindi"]
X, df = build_many(D)
print(f"X={X.shape}  turns={df.group.nunique()}  eot={df.y.sum()}  hold={(1-df.y).sum()}\n")

def lr(C=1.0):
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=2000,
                                                class_weight="balanced"))])

def lr_rank(C=1.0):
    return Pipeline([("qt", QuantileTransformer(n_quantiles=200, output_distribution="normal",
                                                random_state=0)),
                     ("clf", LogisticRegression(C=C, max_iter=2000,
                                                class_weight="balanced"))])

def gb(lr_=0.06, leaves=8, depth=3, l2=1.0):
    return HistGradientBoostingClassifier(
        learning_rate=lr_, max_leaf_nodes=leaves, max_depth=depth, l2_regularization=l2,
        max_iter=300, early_stopping=True, validation_fraction=0.2,
        class_weight="balanced", random_state=0)

print("=== REFERENCE ===")
print("  silence baseline                   english:   1600ms | hindi:    850ms")
print("\n=== MODELS (OOF by turn, real metric, mean±std over 3 seeds) ===")
MODELS = [
    ("logreg C=0.03", lr(0.03)), ("logreg C=0.1", lr(0.1)),
    ("logreg C=0.3", lr(0.3)), ("logreg C=1", lr(1.0)), ("logreg C=3", lr(3.0)),
    ("logreg-rank C=0.3", lr_rank(0.3)), ("logreg-rank C=1", lr_rank(1.0)),
    ("hgb lr.06 leaves8", gb()),
    ("hgb lr.03 leaves4", gb(0.03, 4, 2, 3.0)),
    ("hgb lr.1 leaves16", gb(0.1, 16, 4, 0.3)),
    ("rf depth5", RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=8,
                                         class_weight="balanced", random_state=0, n_jobs=-1)),
]
res = {}
for name, m in MODELS:
    r = evaluate(m, X, df)
    res[name] = r
    report(name, r)

print("\n=== CROSS-LINGUAL TRANSFER (proxy for the mostly-Hindi hidden set) ===")
for name, m in [("logreg C=0.1", lr(0.1)), ("logreg C=0.3", lr(0.3)),
                ("hgb lr.06", gb()), ("hgb lr.03 leaves4", gb(0.03, 4, 2, 3.0))]:
    a = cross_lingual(m, X, df, "english", "hindi")
    b = cross_lingual(m, X, df, "hindi", "english")
    print(f"  {name:22s} en->hi: {a['ms']:6.0f}ms auc={a['auc']:.3f}   "
          f"hi->en: {b['ms']:6.0f}ms auc={b['auc']:.3f}")

print("\n=== FEATURE SIGNAL (univariate AUC on pooled data) ===")
from eotlib.metric import auc_score
from eotlib.features import FEATURE_NAMES
rows = []
for i, n in enumerate(FEATURE_NAMES):
    a_all = auc_score(X[:, i], df.y.values)
    a_hi = auc_score(X[(df.corpus == "hindi").values, i], df.y.values[(df.corpus == "hindi").values])
    a_en = auc_score(X[(df.corpus == "english").values, i], df.y.values[(df.corpus == "english").values])
    rows.append((abs(a_all - .5), n, a_all, a_en, a_hi))
print(f"  {'feature':22s} {'AUC':>6s} {'en':>6s} {'hi':>6s}")
for _, n, a, e, h in sorted(rows, reverse=True):
    print(f"  {n:22s} {a:6.3f} {e:6.3f} {h:6.3f}")
