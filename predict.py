"""Predict p_eot for every pause in a folder.

    python predict.py --data_dir <folder> --out predictions.csv

Works on any folder with the handout's layout (audio/ + labels.csv), including
one it has never seen. Loads the trained artifact -- never refits.

Output: turn_id,pause_index,p_eot

CAUSALITY. For each pause, features come only from audio in [0, pause_start).
`features_for_pause` truncates there and delegates to a function that never
receives the rest of the signal, so nothing downstream can see the future even
by accident. `pause_end` of the pause being scored is future information and is
never read; earlier pauses are complete and in the past, so their timings are
fair game. `tests/test_causality.py` proves this by mutating the post-pause
audio six ways and asserting the features are bit-identical.

This file reads `labels.csv` for pause TIMINGS only. If a `label` column is
present it is ignored -- the run must be identical on a blind labels.csv.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eotlib.data import build                            # noqa: E402

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "artifacts", "model.joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"no model at {args.model} -- run: python train.py")
    art = joblib.load(args.model)

    X, df = build(args.data_dir, n_jobs=args.n_jobs)
    if X.shape[1] != len(art["feature_names"]):
        raise SystemExit(
            f"feature mismatch: artifact has {len(art['feature_names'])} features, "
            f"extractor produced {X.shape[1]}. Retrain: python train.py")

    p = art["model"].predict_proba(X)[:, 1]
    if not np.isfinite(p).all():
        raise SystemExit("non-finite predictions -- refusing to write")

    out = pd.DataFrame({"turn_id": df.turn_id.values,
                        "pause_index": df.pause_index.values,
                        "p_eot": np.round(p, 6)})
    # one row per pause in labels.csv, or the scorer exits with "missing prediction"
    assert len(out) == len(df), f"{len(out)} predictions for {len(df)} pauses"
    assert not out.duplicated(["turn_id", "pause_index"]).any(), "duplicate keys"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} predictions -> {args.out}")
    print(f"  turns={out.turn_id.nunique()}  "
          f"p_eot: min={p.min():.3f} med={np.median(p):.3f} max={p.max():.3f}")


if __name__ == "__main__":
    main()
