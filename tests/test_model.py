"""Guards on the shipped model wrapper and the predict.py contract."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

ROOT = "/Users/vanshikaagarwal/speedrun"
sys.path.insert(0, ROOT)
from eotlib.metric import score_arrays          # noqa: E402
from eotlib.model import RankCalibrated         # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    y = (X[:, 0] + rng.normal(0, 1.5, 200) > 0).astype(int)
    m = RankCalibrated(RandomForestClassifier(n_estimators=50, random_state=0))
    m.fit(X, y)
    return m, X, y


def test_scores_never_fall_below_the_scorers_lowest_threshold(fitted):
    """The whole point of the floor: the scorer sweeps thresholds from 0.05, so
    scores under it delete the silence-baseline fallback from the sweep. We
    measured hindi at 857 ms -- worse than its own 850 ms baseline -- from
    exactly this."""
    m, X, _ = fitted
    p = m.predict_proba(X)[:, 1]
    assert p.min() >= 0.05, f"min score {p.min()} would strand the baseline policy"
    assert p.max() < 1.0


def test_calibration_is_monotone_so_ranking_is_untouched(fitted):
    """Rank calibration must not be able to change AUC -- only which operating
    points the fixed threshold grid can reach."""
    m, X, _ = fitted
    raw = m.base_.predict_proba(X)[:, 1]
    cal = m.predict_proba(X)[:, 1]
    order_raw = np.argsort(np.argsort(raw))
    order_cal = np.argsort(np.argsort(cal))
    # ties may reorder; correlation of ranks must be ~1
    assert np.corrcoef(order_raw, order_cal)[0, 1] > 0.999


def test_never_worse_than_the_silence_baseline():
    """End-to-end property: with the floor in place, the scorer can always fall
    back to 'fire at every pause', so our score is bounded by the baseline."""
    rng = np.random.default_rng(1)
    n_turns = 100
    y, dur, ti = [], [], []
    for t in range(n_turns):
        for k in range(rng.integers(1, 4)):
            y.append(0); dur.append(float(rng.uniform(0.1, 1.5))); ti.append(t)
        y.append(1); dur.append(float(rng.uniform(0.2, 2.0))); ti.append(t)
    y, dur, ti = np.array(y), np.array(dur), np.array(ti)
    base = score_arrays(np.ones(len(y)), y, dur, ti)["latency"]
    # a deliberately useless model, but floored into the legal range
    for seed in range(5):
        p = np.random.default_rng(seed).uniform(0.051, 0.999, len(y))
        got = score_arrays(p, y, dur, ti)["latency"]
        assert got <= base + 1e-9, f"scored {got} vs baseline {base}"


def test_predict_cli_runs_on_an_unseen_folder_and_ignores_labels():
    """predict.py must work on a folder it has never seen, and must not depend
    on the `label` column -- the hidden labels.csv may not carry answers."""
    if not os.path.exists(os.path.join(ROOT, "artifacts", "model.joblib")):
        pytest.skip("model artifact not built")
    src = os.path.join(ROOT, "eot_handout", "eot_data", "hindi")
    with tempfile.TemporaryDirectory() as tmp:
        os.symlink(os.path.join(src, "audio"), os.path.join(tmp, "audio"))
        df = pd.read_csv(os.path.join(src, "labels.csv")).head(12)
        blind = df.drop(columns=["label"])          # labels stripped
        blind.to_csv(os.path.join(tmp, "labels.csv"), index=False)
        out = os.path.join(tmp, "p.csv")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "predict.py"),
                            "--data_dir", tmp, "--out", out],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
        got = pd.read_csv(out)
        assert list(got.columns) == ["turn_id", "pause_index", "p_eot"]
        assert len(got) == len(blind)
        assert got.p_eot.between(0.0, 1.0).all()
        assert got.p_eot.min() >= 0.05
