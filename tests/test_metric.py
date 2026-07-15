"""Verify eotlib.metric.score_arrays is bit-exact vs the official starter/score.py.

The fast scorer drives model selection, so any drift from the official one would
silently corrupt every decision we make. We test against the real scorer on real
prediction files plus random ones (random preds cover operating points that a
trained model never visits, e.g. where nothing meets the 5% budget).
"""
import csv
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = "/Users/vanshikaagarwal/speedrun"
DATA = os.path.join(ROOT, "eot_handout", "eot_data")
sys.path.insert(0, ROOT)
from eotlib.metric import score_arrays  # noqa: E402

# import the official scorer as a module
_spec = importlib.util.spec_from_file_location(
    "official", os.path.join(ROOT, "eot_handout", "starter", "score.py"))
official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(official)


def _load(lang):
    rows = list(csv.DictReader(open(os.path.join(DATA, lang, "labels.csv"))))
    turns = {t: i for i, t in enumerate(sorted({r["turn_id"] for r in rows}))}
    y = np.array([r["label"] == "eot" for r in rows])
    dur = np.array([float(r["pause_end"]) - float(r["pause_start"]) for r in rows])
    ti = np.array([turns[r["turn_id"]] for r in rows])
    return rows, y, dur, ti


def _write_pred(rows, p, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for r, v in zip(rows, p):
            w.writerow([r["turn_id"], r["pause_index"], f"{v:.6f}"])


@pytest.mark.parametrize("lang", ["english", "hindi"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_matches_official_random(lang, seed):
    rows, y, dur, ti = _load(lang)
    rng = np.random.default_rng(seed)
    if seed == 0:
        p = np.ones(len(rows))                 # silence-only baseline
    elif seed == 1:
        p = y * 0.9 + 0.05                     # near-oracle
    else:
        p = rng.random(len(rows))              # arbitrary operating points
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        _write_pred(rows, p, f.name)
    exp = official.score(os.path.join(DATA, lang, "labels.csv"), f.name)
    got = score_arrays(p, y, dur, ti)
    os.unlink(f.name)
    assert got["latency"] == pytest.approx(exp["latency"], abs=1e-9)
    assert got["cutoff"] == pytest.approx(exp["cutoff"], abs=1e-9)
    assert got["threshold"] == pytest.approx(exp["threshold"], abs=1e-9)
    assert got["delay"] == pytest.approx(exp["delay"], abs=1e-9)
    assert got["auc"] == pytest.approx(exp["auc"], abs=1e-9)
    assert got["n_turns"] == exp["n_turns"] and got["n_pauses"] == exp["n_pauses"]


def test_known_reference_points():
    """Guards the numbers we quote in SUMMARY/RUNLOG."""
    for lang, expect in [("english", 1.6), ("hindi", 0.85)]:
        rows, y, dur, ti = _load(lang)
        got = score_arrays(np.ones(len(rows)), y, dur, ti)
        assert got["latency"] == pytest.approx(expect, abs=1e-9)


def test_no_policy_meets_budget_returns_timeout():
    """Degenerate case: every hold is long and always fires -> fall back to 1.6 s."""
    y = np.array([False] * 20 + [True] * 20)
    dur = np.full(40, 3.0)
    ti = np.arange(40) % 20
    got = score_arrays(np.ones(40), y, dur, ti)
    assert got["latency"] == pytest.approx(1.6)
