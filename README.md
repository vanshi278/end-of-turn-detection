# End-of-Turn Detection

Predicts, at every pause in a user turn, the probability that the turn is over.

**Start with [SUMMARY.html](SUMMARY.html)** — solution, results, graphs, and the
human/agent split. [RUNLOG.md](RUNLOG.md) is the scored history (13 runs, wrong
hypotheses included). [NOTES.md](NOTES.md) is the 10-sentence version.

## Results

Out-of-fold (GroupKFold by turn, 3 seeds), scored by the official `score.py`:

| | silence baseline | ours | bootstrap gain (600 turn-resamples) |
|---|---|---|---|
| english | 1600 ms | **1116 ms** | **443 ms** [130, 649], wins 100% |
| hindi | 850 ms | 850 ms | 46 ms [0, 191], wins 50% |

English is a decisive win; **Hindi is not yet a demonstrable one** and we say so.
The same model scores 511 ms in-sample on these folders — that is memorisation
and is not claimed anywhere. See SUMMARY.html § "The honest read on Hindi".

## The data is not in this repo

`eot_data/` is **deliberately not published**. It is real human-to-voice-agent
phone audio — actual customers speaking names, addresses and numbers — and it is
the assignment's proprietary dataset and ground truth. Publishing it would expose
those recordings and hand future candidates the answers. The listening-study
clips cut from it are excluded for the same reason.

To run this, drop the handout in at `eot_handout/eot_data/{english,hindi}/`.
Everything else — code, tests, report, predictions, and the human study's written
results — is here.

## Run it

```bash
python train.py                                            # -> artifacts/model.joblib
python predict.py --data_dir <folder> --out predictions.csv
python eot_handout/starter/score.py --data_dir <folder> --pred predictions.csv

python -m pytest            # fast gate (~100 s)
python -m pytest -m slow    # exhaustive causality: all 496 pauses
python tools/make_summary.py   # regenerates SUMMARY.html from artifacts/results.json
```

`predict.py` runs in well under a minute on a laptop CPU and reads `labels.csv`
for pause *timings* only — a `label` column, if present, is ignored.

## Deliverables

| file | what |
|---|---|
| `SUMMARY.html` | the report (self-contained; inline SVG; light + dark) |
| `predict.py` | `--data_dir <folder> --out predictions.csv` |
| `predictions_english.csv`, `predictions_hindi.csv` | from `predict.py` (in-sample on these folders — see below) |
| `predictions_english_oof.csv`, `predictions_hindi_oof.csv` | **out-of-fold; these are the honest numbers** |
| `RUNLOG.md`, `NOTES.md` | scored history; short notes |

## Layout

```
eotlib/
  features.py   causal prosodic features; truncates at pause_start, then delegates
  yin.py        vectorised YIN pitch tracker + voicing, written from scratch
  metric.py     the official metric, vectorised (bit-exact; see tests)
  cv.py         GroupKFold-by-turn model selection on the REAL metric + hard_auc
  bootstrap.py  paired bootstrap over resampled turns
  model.py      rank calibration + the baseline floor
  data.py       labels + audio -> feature matrix (parallel, cached)
tests/          causality, metric equivalence, YIN validation, model guards
tools/          listening kits, experiments, diagnostics, summary generator
```

## Rules compliance

* **Causality** — features read only `[0, pause_start)`. Enforced structurally
  (the extractor never receives the rest of the signal) and proven by
  `tests/test_causality.py`, which mutates post-pause audio six ways and requires
  bit-identical features. Note the dataset trap: every WAV ends exactly at its eot
  pause, so `len(x)/sr` alone scores AUC 1.000 — we never touch it.
  `dur`/`pause_end` of the scored pause is future information and is used only as
  a *training* weight, from the labels, never as a feature.
* **No pretrained models or downloaded weights.** numpy / scipy / scikit-learn /
  pandas / librosa only. The pitch tracker is our own code; librosa is used for
  framing, RMS, MFCC, and spectral statistics — all deterministic DSP, no weights.
* **Laptop CPU only.** No GPU, no cloud, no external data.
