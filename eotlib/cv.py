"""Model selection against the REAL metric, on held-out TURNS.

Never select on accuracy/AUC. The objective is "mean response delay at <=5%
interrupted turns", and it disagrees with AUC: a model can win AUC by ranking
cheap short holds well, which the metric ignores entirely.

Protocol: GroupKFold by turn (a turn never straddles the split), pool the
out-of-fold predictions, score the pool with the official metric. The pool is
the same size as a real evaluation set (100 turns), so the number means what it
says. Repeated over seeds because with 100 turns the 5% budget is 5 turns and a
single split is noisy.
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .metric import auc_score, score_arrays


def oof_predict(model, X, y, groups, n_splits=5, seed=0, sample_weight=None):
    """Out-of-fold p_eot. Each row scored by a model that never saw its turn."""
    p = np.zeros(len(X), float)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    perm = {g: i for i, g in enumerate(rng.permutation(uniq))}
    shuffled = np.array([perm[g] for g in groups])
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, shuffled):
        m = clone(model)
        if sample_weight is not None:
            m.fit(X[tr], y[tr], **{_sw_key(m): sample_weight[tr]})
        else:
            m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def _sw_key(m):
    return "clf__sample_weight" if hasattr(m, "steps") else "sample_weight"


def hard_auc(p, y, dur, d_ref=0.6):
    """AUC on the ONLY contrast that can change the score: eot vs holds long
    enough to actually interrupt someone (dur > d_ref).

    Plain AUC is measured against every hold, but a hold shorter than the
    agent's delay cannot cost anything -- the user resumes first. Measured on
    this data, a model at overall AUC 0.685 scores 0.52 here: it separates turn
    ends from SHORT holds (worthless) while being at chance on hesitations
    (everything). Report this, not AUC.
    """
    keep = (y == 1) | ((y == 0) & (dur > d_ref))
    if keep.sum() < 10 or len(np.unique(y[keep])) < 2:
        return float("nan")
    return auc_score(np.asarray(p)[keep], np.asarray(y)[keep].astype(bool))


def evaluate(model, X, df, n_splits=5, seeds=(0, 1, 2), sample_weight=None,
             eval_mask=None):
    """-> dict with mean/std of the real metric over seeds, per corpus.

    eval_mask restricts SCORING (not training) -- e.g. score Hindi only.
    """
    y = df.y.values
    groups = df.group.values if "group" in df else df.turn_id.values
    out = {}
    per_seed = {}
    for s in seeds:
        p = oof_predict(model, X, y, groups, n_splits, s, sample_weight)
        for corpus in list(df.corpus.unique()) + ["ALL"] if "corpus" in df else ["ALL"]:
            m = np.ones(len(df), bool) if corpus == "ALL" else (df.corpus == corpus).values
            if eval_mask is not None:
                m &= eval_mask
            if not m.any():
                continue
            g = df.group.values[m] if "group" in df else df.turn_id.values[m]
            _, ti = np.unique(g, return_inverse=True)
            r = score_arrays(p[m], y[m], df.dur.values[m], ti)
            per_seed.setdefault(corpus, []).append(
                (r["latency"], r["auc"], hard_auc(p[m], y[m], df.dur.values[m])))
    for corpus, vals in per_seed.items():
        lat = np.array([v[0] for v in vals]) * 1000
        out[corpus] = {"ms": lat.mean(), "ms_std": lat.std(),
                       "auc": np.mean([v[1] for v in vals]),
                       "hauc": np.nanmean([v[2] for v in vals])}
    return out


def cross_lingual(model, X, df, train_corpus, test_corpus, sample_weight=None):
    """Train on one language, score the other. Proxy for the unseen-Hindi test."""
    tr = (df.corpus == train_corpus).values
    te = (df.corpus == test_corpus).values
    m = clone(model)
    if sample_weight is not None:
        m.fit(X[tr], df.y.values[tr], **{_sw_key(m): sample_weight[tr]})
    else:
        m.fit(X[tr], df.y.values[tr])
    p = m.predict_proba(X[te])[:, 1]
    _, ti = np.unique(df.group.values[te], return_inverse=True)
    r = score_arrays(p, df.y.values[te], df.dur.values[te], ti)
    return {"ms": r["latency"] * 1000, "auc": r["auc"]}


def report(name, res):
    parts = [f"{c}: {v['ms']:6.0f}±{v['ms_std']:3.0f}ms hard_auc={v['hauc']:.3f}"
             for c, v in sorted(res.items())]
    print(f"  {name:34s} " + " | ".join(parts))
