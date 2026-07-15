"""Write OUT-OF-FOLD predictions for the provided folders, and score them with
the official scorer.

Why this file exists. The shipped model is trained on both provided languages --
correct, because the hidden set is different turns and we want all 200 of ours.
But that makes `predict.py`'s output on those same folders IN-SAMPLE: it scores
511 ms at AUC 0.94, versus 0.71 out-of-fold. Quoting 511 ms would be reporting a
memorisation score.

Here every prediction comes from a model that never saw that turn (GroupKFold by
turn), so feeding these to the official scorer gives an honest estimate of what
the hidden set will do. These are the numbers in RUNLOG.md.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/vanshikaagarwal/speedrun")
from eotlib.cv import oof_predict                        # noqa: E402
from eotlib.data import build_many                       # noqa: E402
from train import build_model, cost_weights, DEFAULT_DIRS  # noqa: E402

X, df = build_many(DEFAULT_DIRS)
y, dur = df.y.values, df.dur.values

# average over seeds: with 100 turns the 5% budget is 5 turns, so a single
# fold assignment is noisy
ps = [oof_predict(build_model(), X, y, df.group.values, seed=s,
                  sample_weight=cost_weights(y, dur)) for s in (0, 1, 2)]
p = np.mean(ps, axis=0)

for d in DEFAULT_DIRS:
    corpus = os.path.basename(os.path.normpath(d))
    m = (df.corpus == corpus).values
    out = pd.DataFrame({"turn_id": df.turn_id.values[m],
                        "pause_index": df.pause_index.values[m],
                        "p_eot": np.round(p[m], 6)})
    path = f"predictions_{corpus}_oof.csv"
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} out-of-fold predictions -> {path}")
