# RUNLOG

Score = mean response delay at <= 5% interrupted turns (lower is better), from
the official `starter/score.py`.

**Every score below is out-of-fold** (GroupKFold by turn, averaged over 3 seeds):
each pause is scored by a model that never saw its turn. In-sample numbers on
these folders are meaningless and we do not quote them -- see run 11, where the
shipped model scores 511 ms in-sample against 1116 ms out-of-fold.

`hard_auc` is our own diagnostic, introduced in run 6: AUC on **eot vs holds
longer than 0.6 s** -- the only pauses that can change the score. Plain AUC is
measured against every hold, including the ~75% that are too short to ever
interrupt anyone.

| system | english | hindi |
|---|---|---|
| silence-only baseline (given) | 1600 ms | 850 ms |
| starter `train.py` | 1190 ms | 850 ms |
| **ours (out-of-fold)** | **1116 ms** | **850 ms** |

---

### Run 1 - measure the ground truth before touching anything
`baseline.py` + `starter/train.py`, both languages.
**english 1600 / hindi 850** (baseline); starter 1190 / 850.
Hindi's baseline is already strong because its holds are short (median 400 ms,
p90 630 ms): a silence timer at 850 ms rarely interrupts. The hidden set is
*mostly Hindi*, so **850 ms is the real bar**, not the 1600 ms headline. Logged
so we never mistake an English-only win for a win.

### Run 2 - leakage audit (no model)
Every WAV ends exactly at its eot pause: audio after the pause is 0.000 s for all
200 eots vs a mean of 6.7 s for holds. A one-line model on `len(x)/sr` scores
**AUC 1.000 / 100 ms (english), 242 ms (hindi)**.
That is the trap in this dataset and it is worthless -- the brief says the feature
code will be read. Response: `features_for_pause` truncates at pause_start and
delegates to a function that never receives the full signal, so leaking is not
possible rather than merely discouraged. `tests/test_causality.py` mutates
post-pause audio six ways (append noise / silence / a tone / reversed audio,
zero it, truncate) and asserts features are **bit-identical**.

### Run 3 - first feature set (25 causal prosodic features) + logistic regression
**english 1284 / hindi 850** (AUC 0.68).
English beats baseline; Hindi does not move *at all* -- every model returns
exactly 850±0. A real model should at least wobble. Flagged as suspicious.

### Run 4 - cost-aware sample weighting
A hold only causes a false cutoff if it outlives the agent's delay; at a 0.65 s
delay **90% of Hindi holds are already free**. So weight each hold by whether it
can actually cost anything. (`dur` is used at TRAIN time only, from the labels,
exactly like `class_weight`. It is future information at inference and never
reaches a feature.)
**english 1284 -> 1150 / hindi 850.**
Overall **AUC got worse (0.683 -> 0.664) while the score got better** -- the first
hard evidence that AUC is the wrong objective here.

### Run 5 - fixed the pitch tracker; wrote our own
Voicing was gated on "is the YIN value in range", which called **99% of frames
voiced**. Voiced-run segmentation was therefore meaningless, and with it final
lengthening -- the strongest turn-end cue in the literature (`voiced_run_final_s`
scored AUC 0.517, i.e. noise).
`librosa.pyin` fixes voicing but costs 4.7 s/pause: ~41 CPU-min to featurise the
corpus and ~20 min for `predict.py` on the grader's laptop. Not shippable. So we
wrote a **vectorised YIN from scratch** (`eotlib/yin.py`), using the CMND minimum
as a principled aperiodicity/voicing estimate instead of pyin's Viterbi.
Validated (`tests/test_yin.py`): recovers synthetic pitch to <2%, rejects noise
and silence, no octave errors, **85% voicing agreement with pyin, 0.42% median f0
disagreement over 1832 real frames, 68x faster** (40 ms vs 2731 ms per 10 s
window). Corpus featurisation 3 min -> 2.5 s; per-pause cost 202 ms, so
`predict.py` finishes in under a minute.
Also frame-local by construction, so it cannot smooth across the pause boundary.
*(Aside: the pyin comparison test initially failed at 20% disagreement. The cause
was our test, not our code -- it averaged per-file and one near-silent clip, where
pyin had railed against its own 400 Hz ceiling on 8 frames, dominated the mean.
Pooling frames and dropping railed references fixed it.)*
**english 1215 / hindi 850** (AUC 0.686). Better tracker, wrong target.

### Run 6 - why Hindi will not move: the headline AUC is fake
AUC on **eot vs holds long enough to matter**:

| delay | overall AUC | eot-vs-long-hold AUC | n long holds |
|---|---|---|---|
| 0.55 s | 0.685 | **0.567** | 25 |
| 0.65 s | 0.685 | **0.521** | 15 |

The model separates turn ends from *short* holds -- which the metric does not
score -- and is **at chance on hesitations**, which are everything. By language:
english hard_auc **0.704** (real), hindi **0.509** (chance). Same features, same
model, same data volume.
Promoted `hard_auc` to the headline diagnostic so this cannot hide behind AUC
again. The gap, precisely: at a 0.65 s delay we fire on 9 risky turns against a
budget of 5, and the entire decision rests on ~15 pause instances.

### Run 7 - HUMAN: blind listening study (36 clips, balanced, Hindi-weighted)
Kit deliberately samples **long holds only** (>=0.55 s), so it measures the exact
contrast the model is at chance on. Human (native Hindi speaker), blind, hearing
only what the model hears:
**72.2% overall; 75.0% Hindi (n=24); 87.5% on confident Hindi calls.**
Confidence is calibrated (conf-5 82%, conf-4 80%, conf-3 67%), so the uncertainty
tracks a real, audible property.
**Verdict: the Hindi signal is audible and our features were blind to it** --
~24 points of headroom, not a data ceiling. This is why we kept going.

### Run 8 - features built from the human's notes
Three families, each traceable to a specific note (see SUMMARY.html):
* **pitch position within the speaker's own range** ("pitch dropped" -> eot;
  "pitch was very high, cant be the end" -> hold). Encoded as a *percentile* of
  that speaker's own pitch history, not a z-score -- the ear asks "did you come
  home to your floor", not "how many IQRs from your median". Beat the z-score
  head to head: `f0_final_pct` 0.639 vs `f0_final_z` 0.595 (hindi hard_auc).
* **breath / trailing aspiration** ("person taking breath" -> hold).
* **final-nucleus vowel identity via CMN'd MFCCs** -- the human read Hindi verb
  endings lexically ("rahega used", "ended with rakhiyega"). We cannot do ASR,
  but Hindi is verb-final and its finite verbs land on a small set of vowel
  qualities, which MFCCs encode.

Hindi univariate hard_auc jumped from ~0.52 (whole model) to **mfcc7 0.703,
mfcc5 0.696, unvoiced_frac_final 0.650, abruptness 0.616** as *single features*.
Model: **hindi hard_auc 0.509 -> 0.707**, english -> 0.715. ALL 1078 ms.

### Run 9 - rank calibration (hypothesis half-wrong, kept anyway)
The scorer sweeps a fixed threshold grid, so the reachable operating points
depend on the *shape* of the score distribution. Our RF's probabilities spanned
only 0.225..0.782 -- **just 11 of 19 thresholds split the data at all**.
Hypothesis: spreading scores to uniform unlocks Hindi. **It did not** -- at a
0.65 s delay we were already cutting exactly 5 turns, the maximum allowed, so the
constraint was binding and no threshold placement could help. Kept regardless: it
is free, monotone (AUC provably unchanged), and helped english 1184 -> 1150.
Recorded as a wrong hypothesis rather than quietly dropped.

### Run 10 - the plateau is features, not model
Swept complexity (rf depths 3/4/5, ExtraTrees, logreg C=0.03..1) against OOF
hard_auc + paired bootstrap: everything lands at hindi hard_auc 0.61-0.71.
Best: `rf max_depth=4, min_samples_leaf=12, max_features=0.3` (hindi 0.710).
The ceiling is what we can hear, not what we can fit.

### Run 11 - the in-sample trap, caught
`predict.py` on the provided folders scored **511 ms, AUC 0.92-0.94**. It is
in-sample -- the shipped model trained on those exact turns (0.94 in-sample vs
0.71 OOF). Quoting it would be reporting memorisation, the same species of error
as the run-2 file-length leak. Added `tools/make_oof_predictions.py` so every
number we publish comes from models that never saw the turn they score.
**Honest: english 1116 / hindi 850.**

### Run 12 - never lose to the baseline (a real bug)
Out-of-fold Hindi came out at **857 ms -- worse than its own 850 ms baseline**.
Cause: the scorer's threshold grid *starts* at 0.05, so any score below 0.05
deletes the "fire at every pause" policy from the sweep. A few turn ends scored
under it and took the baseline fallback with them.
Fix: floor calibrated scores at 0.051 (`RankCalibrated.P_FLOOR`). Monotone, so
ranking and AUC are untouched, and it buys a guarantee: **the sweep can always
find the baseline, so we can never score worse than it.** Pinned by
`tests/test_model.py::test_never_worse_than_the_silence_baseline`.
**english 1116 (from 1134) / hindi 850.**

### Run 13 - FINAL, and what it honestly means
Paired bootstrap over resampled turns (600 draws) -- because a single score on a
single sample cannot tell "no better than a timer" from "better, but this sample's
baseline is degenerate". Hindi's 850 is exactly such a degeneracy: exactly 5 of
its 100 turns contain a hold >0.85 s, which is exactly the 5% budget, so "fire at
everything" saturates the constraint. That is a property of this draw, not of
Hindi.

| | baseline | ours | gain | wins |
|---|---|---|---|---|
| english | 1556 ms [1300, 1600] | **1113 ms** [938, 1405] | **443 ms** [130, 649] | **100%** |
| hindi | 882 ms [750, 1000] | **836 ms** [659, 1000] | 46 ms [0, 191] | 50% |

**English is a decisive win. Hindi is not yet a demonstrable one** -- the mean
improves and the interval touches zero. We report it that way. hard_auc 0.71 says
the discrimination is real; converting it into milliseconds needs either a lower
delay (more precision on long holds) or a hidden set whose baseline is less
degenerate than this one's.

Final: **english 1116 ms / hindi 850 ms**, out-of-fold, official scorer.
