"""Train the end-of-turn model and save a versioned artifact.

    python train.py                       # both provided languages
    python train.py --data_dir a --data_dir b --out artifacts/model.joblib

predict.py LOADS this artifact. It never refits -- the hidden set is scored by a
model that has never seen those turns, so training must happen exactly once,
here.

Design decisions and the evidence for each are in RUNLOG.md. The short version:
  * one multilingual model, not one per language -- the hidden set is "mostly
    Hindi" but we cannot detect language at inference without guessing from a
    filename, and pooling doubles our 200 turns
  * cost-aware sample weights: a hold that is shorter than the agent's delay
    cannot cause a false cutoff, so it should not consume model capacity
  * rank calibration: the scorer sweeps a fixed threshold grid, so the shape of
    the score distribution decides which operating points are reachable
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eotlib.data import build_many                      # noqa: E402
from eotlib.features import FEATURE_NAMES               # noqa: E402
from eotlib.model import RankCalibrated                 # noqa: E402

FEATURE_VERSION = 4          # bump whenever features.py changes meaningfully
DEFAULT_DIRS = ["eot_handout/eot_data/english", "eot_handout/eot_data/hindi"]

# Cost weighting. A hold only causes a false cutoff if it outlives the agent's
# action delay; at a 0.6 s delay ~90% of Hindi holds are already free. `dur`
# comes from the training labels and is used ONLY here, exactly like
# class_weight. It is future information at inference and never reaches a
# feature -- see eotlib/features.py and tests/test_causality.py.
COST_D_REF = 0.60
COST_SOFT = 0.15
COST_W_MIN = 0.05


def cost_weights(y, dur):
    w = np.ones(len(y), float)
    h = y == 0
    w[h] = COST_W_MIN + (1 - COST_W_MIN) / (1 + np.exp(-(dur[h] - COST_D_REF) / COST_SOFT))
    return w


def build_model():
    """Shallow, leaf-constrained, feature-subsampled forest.

    496 pauses and 51 features is a small-data problem: an unconstrained forest
    hits in-sample AUC 0.94 against 0.71 out-of-fold, i.e. it memorises. Chosen
    by out-of-fold hard_auc + paired bootstrap, never by in-sample fit
    (see RUNLOG run 10). Rank calibration on top so the scorer's fixed threshold
    grid can actually address the ranking (eotlib/model.py).
    """
    return RankCalibrated(RandomForestClassifier(
        n_estimators=800, max_depth=4, min_samples_leaf=12, max_features=0.3,
        class_weight="balanced", random_state=0, n_jobs=-1))


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", action="append", default=None,
                    help="repeatable; defaults to both provided languages")
    ap.add_argument("--out", default="artifacts/model.joblib")
    args = ap.parse_args()
    dirs = args.data_dir or DEFAULT_DIRS

    X, df = build_many(dirs)
    y, dur = df.y.values, df.dur.values
    print(f"training on {X.shape[0]} pauses / {df.group.nunique()} turns "
          f"from {len(dirs)} corpora, {X.shape[1]} features")
    print(f"  eot={int(y.sum())}  hold={int((1-y).sum())}  "
          f"holds that can actually cost us (dur>{COST_D_REF}s)="
          f"{int(((y==0)&(dur>COST_D_REF)).sum())}")

    model = build_model()
    model.fit(X, y, sample_weight=cost_weights(y, dur))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "trained_on": dirs,
        "n_train_pauses": int(len(y)),
        "n_train_turns": int(df.group.nunique()),
        "cost_weighting": {"d_ref": COST_D_REF, "soft": COST_SOFT, "w_min": COST_W_MIN},
        "sklearn_version": __import__("sklearn").__version__,
        "numpy_version": np.__version__,
        "git_rev": _git_rev(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, args.out)
    print(f"wrote {args.out}")
    print(json.dumps({"feature_version": FEATURE_VERSION,
                      "n_features": len(FEATURE_NAMES)}, indent=None))


if __name__ == "__main__":
    main()
