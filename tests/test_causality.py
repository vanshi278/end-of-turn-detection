"""Prove the features cannot see the future.

The assignment says: "for a pause, your features may use ONLY audio from time 0
up to pause_start of that pause... We will read your feature code to check this."

Reading code catches obvious violations. These tests catch subtle ones -- a
pitch tracker that smooths across the pause boundary, a normaliser fitted on the
whole file, an off-by-one that leaks one frame of the future.

Method: take every real pause, then MUTATE everything after pause_start --
append noise, truncate, zero it, replace with speech from another turn. If a
single feature moves, information crossed the boundary. Features must be
BIT-IDENTICAL, not merely close.

This also guards the dataset's built-in trap: every WAV ends exactly at its eot
pause, so file length alone separates the classes perfectly (measured: AUC
1.000, 100 ms on English). test_file_length_is_not_a_feature pins that shut.
"""
import functools
import os
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT = "/Users/vanshikaagarwal/speedrun"
DATA = os.path.join(ROOT, "eot_handout", "eot_data")
sys.path.insert(0, ROOT)
from eotlib.features import features_for_pause  # noqa: E402

RNG = np.random.default_rng(0)


@functools.lru_cache(maxsize=1)
def _all_cases():
    """Lazy: building this eagerly made every xdist worker read 200 WAVs at
    import time, even for the 20 cases the fast tier actually runs."""
    out = []
    for lang in ("english", "hindi"):
        df = pd.read_csv(os.path.join(DATA, lang, "labels.csv"))
        for _, g in df.groupby("turn_id"):
            g = g.sort_values("pause_index")
            x, sr = sf.read(os.path.join(DATA, lang, g.iloc[0].audio_file), dtype="float32")
            prior = []
            for _, r in g.iterrows():
                out.append((x, sr, float(r.pause_start), list(prior)))
                prior.append((float(r.pause_start), float(r.pause_end)))
    return out


def _n_pauses():
    """Case count without touching a single audio file."""
    return sum(len(pd.read_csv(os.path.join(DATA, l, "labels.csv")))
               for l in ("english", "hindi"))


N_ALL = _n_pauses()
# Two tiers. pYIN costs ~1.5 s per pause and each case runs 6 mutations, so the
# exhaustive sweep is minutes. The fast tier is the everyday gate; the full
# sweep is opt-in (`pytest -m slow`) and is what runs before shipping.
FAST_IDX = list(range(0, N_ALL, max(1, N_ALL // 20)))


def _mutations(x, sr, n):
    """Every way we can think of to smuggle the future into the features."""
    return {
        "truncated_at_pause": x[:n],
        "silence_appended": np.concatenate([x[:n], np.zeros(10 * sr, np.float32)]),
        "noise_appended": np.concatenate([x[:n], RNG.normal(0, .1, 10 * sr).astype(np.float32)]),
        "future_zeroed": np.concatenate([x[:n], np.zeros(max(0, len(x) - n), np.float32)]),
        "future_reversed": np.concatenate([x[:n], x[:n][::-1]]),
        "loud_tone_appended": np.concatenate([
            x[:n], (0.5 * np.sin(2 * np.pi * 220 * np.arange(5 * sr) / sr)).astype(np.float32)]),
    }


def _assert_causal(case):
    x, sr, ps, prior = case
    base = features_for_pause(x, sr, ps, prior)
    n = int(round(ps * sr))
    for name, xm in _mutations(x, sr, n).items():
        got = features_for_pause(xm, sr, ps, prior)
        bad = np.flatnonzero(got != base)
        assert not len(bad), (
            f"{name}: future audio changed features at indices {bad.tolist()} "
            f"-- CAUSALITY VIOLATION")


@pytest.mark.parametrize("i", FAST_IDX, ids=lambda i: f"pause{i}")
def test_future_audio_cannot_change_features(i):
    _assert_causal(_all_cases()[i])


@pytest.mark.slow
@pytest.mark.parametrize("i", range(N_ALL), ids=lambda i: f"all{i}")
def test_future_audio_cannot_change_features_every_pause(i):
    """Exhaustive: every pause in both corpora. Run before shipping."""
    _assert_causal(_all_cases()[i])


def test_file_length_is_not_a_feature():
    """The dataset's trap: file length alone is a perfect classifier.

    Same pause, wildly different file lengths -> identical features.
    """
    x, sr, ps, prior = _all_cases()[0]
    n = int(round(ps * sr))
    a = features_for_pause(np.concatenate([x[:n], np.zeros(1 * sr, np.float32)]), sr, ps, prior)
    b = features_for_pause(np.concatenate([x[:n], np.zeros(30 * sr, np.float32)]), sr, ps, prior)
    assert np.array_equal(a, b)


def test_current_pause_end_is_not_consumed():
    """features_for_pause has no pause_end parameter and must not accept one."""
    import inspect
    sig = inspect.signature(features_for_pause)
    assert "pause_end" not in sig.parameters
    assert "pause_duration" not in sig.parameters


def test_prior_pauses_after_pause_start_are_ignored():
    """Defensive: if a caller leaks future pauses in, we must drop them."""
    x, sr, ps, prior = _all_cases()[3]
    clean = features_for_pause(x, sr, ps, prior)
    poisoned = features_for_pause(x, sr, ps, list(prior) + [(ps + 5.0, ps + 7.0)])
    assert np.array_equal(clean, poisoned)


def test_features_are_deterministic():
    x, sr, ps, prior = _all_cases()[7]
    assert np.array_equal(features_for_pause(x, sr, ps, prior),
                          features_for_pause(x, sr, ps, prior))


def test_no_nan_or_inf_anywhere():
    for i in FAST_IDX:
        x, sr, ps, prior = _all_cases()[i]
        f = features_for_pause(x, sr, ps, prior)
        assert np.isfinite(f).all()
