"""How far are we from breaking the Hindi 850 ms wall, and what exactly is missing?"""
import sys
import numpy as np
sys.path.insert(0, "/Users/vanshikaagarwal/speedrun")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eotlib.data import build_many
from eotlib.cv import oof_predict
from eotlib.metric import auc_score

X, df = build_many(["eot_handout/eot_data/english", "eot_handout/eot_data/hindi"])
m = Pipeline([("sc", StandardScaler()),
              ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))])
p = oof_predict(m, X, df.y.values, df.group.values, seed=0)

hi = (df.corpus == "hindi").values
d_, p_ = df[hi], p[hi]
y, dur, grp = d_.y.values, d_.dur.values, d_.group.values
n_turns = len(np.unique(grp))

print("HINDI: what the scorer sees, at each action delay\n")
print(f"{'delay':>6s} {'at-risk turns':>13s} {'recall needed':>13s} "
      f"{'best thresh':>11s} {'recall got':>10s} {'turns cut':>9s} {'latency':>9s}")
for d in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
    risky = np.unique(grp[(y == 0) & (dur > d)])
    need = (1.6 - 0.850) / (1.6 - d)
    best = None
    for t in np.round(np.arange(0.05, 1.0, 0.05), 3):
        fires = p_ >= t
        cut = len(np.unique(grp[fires & (y == 0) & (dur > d)]))
        rec = fires[y == 1].mean()
        lat = rec * d + (1 - rec) * 1.6
        if cut / n_turns <= 0.05 and (best is None or lat < best[0]):
            best = (lat, t, rec, cut)
    if best:
        lat, t, rec, cut = best
        flag = "  <-- BEATS 850" if lat < 0.850 else ""
        print(f"{d:6.2f} {len(risky):13d} {need:12.1%} {t:11.2f} {rec:10.1%} "
              f"{cut:9d} {lat*1000:8.0f}ms{flag}")
    else:
        print(f"{d:6.2f} {len(risky):13d} {need:12.1%} {'--':>11s} "
              f"{'--':>10s} {'--':>9s}  no policy fits budget")

print("\nThe gap, stated precisely:")
for d in [0.55, 0.65]:
    need = (1.6 - 0.850) / (1.6 - d)
    long_hold = (y == 0) & (dur > d)
    sub = y[y == 1].tolist() + [0] * int(long_hold.sum())
    sc = np.concatenate([p_[y == 1], p_[long_hold]])
    a = auc_score(sc, np.array(sub, bool))
    # threshold that hits the required recall; how many risky turns fire there?
    t = np.quantile(p_[y == 1], 1 - need)
    fired_turns = len(np.unique(grp[(p_ >= t) & long_hold]))
    print(f"  delay {d}s: need {need:.0%} eot recall -> threshold {t:.3f}")
    print(f"    at that threshold we fire on {fired_turns} risky turns; budget is 5")
    print(f"    eot-vs-long-hold AUC = {a:.3f}  (n_long_holds={int(long_hold.sum())})")
