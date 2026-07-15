"""Bootstrap the metric over turns.

Why this is not optional here. The Hindi score is pinned at exactly 850 ms
because exactly 5 of its 100 turns contain a hold longer than 0.85 s -- which is
exactly the 5% budget, so "fire at everything after 850 ms" saturates the
constraint and nothing else can win. That is a coincidence of THIS sample of 100
turns, not a property of Hindi. The graded set is different turns, where the
alignment will land somewhere else.

A single number on a single sample cannot distinguish "our model is no better
than a silence timer" from "our model is better, but this particular sample has
a degenerate baseline". Resampling turns with replacement answers the question
the hidden test set actually asks: on a FRESH draw of turns from this
distribution, how often, and by how much, do we beat the baseline?

Turn-level resampling (not pause-level) because a turn is the unit the metric
counts: a turn is interrupted if ANY of its holds fires.
"""
import numpy as np

from .metric import score_arrays


def bootstrap_score(p, y, dur, groups, n_boot=400, seed=0, budget=0.05):
    """-> array of scores (ms), one per bootstrap resample of turns."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_turn = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows, ti = [], []
        for k, g in enumerate(pick):
            ii = idx_by_turn[g]
            rows.append(ii)
            ti.append(np.full(len(ii), k))       # resampled turns are distinct
        rows, ti = np.concatenate(rows), np.concatenate(ti)
        r = score_arrays(p[rows], y[rows], dur[rows], ti, budget=budget)
        out.append(r["latency"] * 1000)
    return np.array(out)


def compare_to_baseline(p, y, dur, groups, n_boot=400, seed=0):
    """Paired bootstrap: our scores vs the silence-only baseline (p=1 always),
    on the SAME resamples. Paired, because sample-to-sample variation in the
    baseline is exactly the noise we are trying to see past.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_turn = {g: np.flatnonzero(groups == g) for g in uniq}
    ours, base = [], []
    ones = np.ones(len(p))
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows, ti = [], []
        for k, g in enumerate(pick):
            ii = idx_by_turn[g]
            rows.append(ii)
            ti.append(np.full(len(ii), k))
        rows, ti = np.concatenate(rows), np.concatenate(ti)
        ours.append(score_arrays(p[rows], y[rows], dur[rows], ti)["latency"] * 1000)
        base.append(score_arrays(ones[rows], y[rows], dur[rows], ti)["latency"] * 1000)
    ours, base = np.array(ours), np.array(base)
    d = base - ours                      # positive == we are faster
    return {
        "ours_mean": ours.mean(), "ours_lo": np.percentile(ours, 2.5),
        "ours_hi": np.percentile(ours, 97.5),
        "base_mean": base.mean(), "base_lo": np.percentile(base, 2.5),
        "base_hi": np.percentile(base, 97.5),
        "gain_mean": d.mean(), "gain_lo": np.percentile(d, 2.5),
        "gain_hi": np.percentile(d, 97.5),
        "win_rate": float((d > 0).mean()),
    }
