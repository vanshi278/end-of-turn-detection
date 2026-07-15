"""Model wrappers.

RankCalibrated exists because of a property of the scorer that is easy to miss:
it sweeps a FIXED threshold grid (0.05, 0.10, ... 0.95). The operating points
you can actually reach therefore depend on the SHAPE of your score
distribution, not just on its ranking.

Measured on Hindi: a random forest's probabilities span only 0.225..0.782, so
just 11 of the 19 thresholds split the data at all, and they land at uneven,
arbitrary quantiles. The model had the ranking to hit 79% recall at a 0.65 s
delay but could only reach 75% -- there was no threshold in the right place.
The information was there; the grid could not address it.

Mapping scores through their own empirical CDF makes the distribution uniform,
so every threshold t lands exactly at the t-th quantile: 19 evenly spaced,
maximally informative operating points.

This is a monotone transform, so it cannot change AUC or the ranking -- it only
changes which (recall, cutoff) trade-offs the scorer is able to find. The CDF is
fitted on TRAINING predictions only and applied per-pause at inference, so it is
plain calibration: no peeking at the test set, no transductive ranking across
turns, and nothing a live agent could not do with a lookup table.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class RankCalibrated(BaseEstimator, ClassifierMixin):
    """Wrap a classifier; push predict_proba through the train-set CDF."""

    # Must sit strictly above the scorer's lowest swept threshold (0.05) so the
    # "fire at every pause" baseline policy always stays reachable. See _cdf.
    P_FLOOR = 0.051

    def __init__(self, base=None, n_grid=512):
        self.base = base
        self.n_grid = n_grid

    def fit(self, X, y, sample_weight=None):
        self.base_ = clone(self.base)
        if sample_weight is not None:
            self.base_.fit(X, y, **{self._sw_key(self.base_): sample_weight})
        else:
            self.base_.fit(X, y)
        p = self.base_.predict_proba(X)[:, 1]
        # empirical CDF of TRAIN scores, stored as a lookup grid
        self.knots_ = np.quantile(p, np.linspace(0, 1, self.n_grid))
        self.knots_ = np.maximum.accumulate(self.knots_)  # enforce monotone
        self.classes_ = self.base_.classes_
        return self

    @staticmethod
    def _sw_key(m):
        return "clf__sample_weight" if hasattr(m, "steps") else "sample_weight"

    def _cdf(self, p):
        # fraction of train scores below p -> uniform on [0,1]
        r = np.searchsorted(self.knots_, p, side="left") / (len(self.knots_) - 1)
        # Then squeeze into [P_FLOOR, 1). This is a safety net, not cosmetics.
        # The scorer's threshold grid STARTS at 0.05, so the policy "fire at
        # every pause" -- the silence-only baseline -- is only reachable if
        # every score is >= 0.05. Emit anything below and that fallback vanishes
        # from the sweep: we measured hindi at 857 ms, WORSE than its own 850 ms
        # baseline, purely because a few turn ends scored under 0.05. With the
        # floor in place the sweep can always find the baseline, so our score is
        # bounded by it and can only improve on it. The map is monotone, so
        # ranking and AUC are untouched.
        return self.P_FLOOR + (1.0 - self.P_FLOOR - 1e-4) * np.clip(r, 0.0, 1.0)

    def predict_proba(self, X):
        p = self.base_.predict_proba(X)[:, 1]
        q = self._cdf(p)
        return np.column_stack([1 - q, q])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
