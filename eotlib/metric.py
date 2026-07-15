"""The official metric, reimplemented in-process and vectorised.

Why this file exists: `starter/score.py` only runs from the command line on a
CSV. To *select* models we need the real metric inside a cross-validation loop,
thousands of times. Selecting on AUC or accuracy instead is the classic mistake
here -- they are not the objective and they disagree with it (a model can gain
AUC by better ranking cheap short holds, which the metric does not care about
at all).

`score_arrays` is verified bit-exact against starter/score.py in tests/.

Metric recap
------------
The agent waits `delay` seconds of silence at each pause; if p_eot >= `threshold`
it takes the floor. Sweep (threshold x delay), report the lowest mean response
delay whose interrupted-turn rate stays <= 5%.

Two structural facts drive the whole model design:
  1. A `hold` pause only costs anything if `delay < pause_duration` -- if the
     user resumes before the agent's timer fires, nobody notices. Short holds
     are FREE.
  2. Cost is counted per TURN, not per pause: a turn is "interrupted" if ANY of
     its holds fires. The 2nd mistake in a turn is free.
"""
import numpy as np

TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)


def score_arrays(p, y, dur, turn_idx, budget=0.05,
                 thresholds=THRESHOLDS, delays=DELAYS):
    """Vectorised sweep. Returns dict matching starter/score.py's `score`.

    p        (n,) predicted p_eot
    y        (n,) 1 = eot, 0 = hold
    dur      (n,) pause duration in seconds (label info; TRAIN/EVAL ONLY --
                  never a feature, it is future information at inference)
    turn_idx (n,) integer turn id, contiguous from 0
    """
    p = np.asarray(p, float)
    y = np.asarray(y, bool)
    dur = np.asarray(dur, float)
    turn_idx = np.asarray(turn_idx, int)
    n_turns = turn_idx.max() + 1 if len(turn_idx) else 1

    fires = p[None, :] >= thresholds[:, None]          # (T, n)
    hold, eot = ~y, y

    # ---- interrupted-turn rate, for every (threshold, delay) ----
    # a hold pause cuts the user iff it fires AND the agent's timer beats them
    cuts = fires[:, None, :] & hold[None, None, :] & (delays[None, :, None] < dur[None, None, :])
    cut_rate = np.zeros((len(thresholds), len(delays)))
    for t in range(len(thresholds)):
        for d in range(len(delays)):
            hit = cuts[t, d]
            cut_rate[t, d] = len(np.unique(turn_idx[hit])) / max(1, n_turns)

    # ---- mean response delay on the true turn ends ----
    # fired -> we answer after `delay`; missed -> the 1.6 s timeout saves us
    recall = (fires[:, eot].sum(axis=1) / max(1, eot.sum()))        # (T,)
    latency = recall[:, None] * delays[None, :] + (1 - recall)[:, None] * TIMEOUT_S

    ok = cut_rate <= budget
    if not ok.any():
        return {"latency": TIMEOUT_S, "cutoff": 0.0, "threshold": 1.0,
                "delay": TIMEOUT_S, "auc": auc_score(p, y), "n_turns": int(n_turns),
                "n_pauses": int(len(p))}
    masked = np.where(ok, latency, np.inf)
    ti, di = np.unravel_index(np.argmin(masked), masked.shape)
    return {"latency": float(latency[ti, di]), "cutoff": float(cut_rate[ti, di]),
            "threshold": float(thresholds[ti]), "delay": float(delays[di]),
            "auc": auc_score(p, y), "n_turns": int(n_turns), "n_pauses": int(len(p))}


def auc_score(p, y):
    """Diagnostic only -- identical to the scorer's rank AUC. NOT the objective."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    order = np.argsort(np.asarray(p, float))
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
