"""Validate the hand-written YIN against synthetic ground truth and pyin.

A bespoke pitch tracker is exactly the kind of code that is subtly wrong and
silently poisons every feature downstream, so it gets checked against:
  1. synthetic signals with a KNOWN f0 (absolute correctness)
  2. librosa.pyin on real speech (agreement with a trusted implementation)
"""
import os
import sys

import numpy as np
import pytest
import soundfile as sf

ROOT = "/Users/vanshikaagarwal/speedrun"
sys.path.insert(0, ROOT)
from eotlib.yin import yin, voiced_probability  # noqa: E402

SR = 16000


def _tone(f0, dur=1.0, sr=SR, harmonics=5, jitter=0.0):
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    for h in range(1, harmonics + 1):        # glottal-ish: decaying harmonics
        x += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    if jitter:
        x += jitter * np.random.default_rng(0).normal(0, 1, len(t))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


@pytest.mark.parametrize("f0_true", [80, 110, 150, 200, 260, 330])
def test_recovers_known_pitch(f0_true):
    f0, voiced, _ = yin(_tone(f0_true), SR)
    assert voiced.mean() > 0.9, f"only {voiced.mean():.0%} of a pure tone marked voiced"
    est = np.median(f0[voiced])
    assert abs(est - f0_true) / f0_true < 0.02, f"got {est:.1f} Hz, want {f0_true}"


@pytest.mark.parametrize("f0_true", [90, 180, 300])
def test_robust_to_noise(f0_true):
    f0, voiced, _ = yin(_tone(f0_true, jitter=0.15), SR)
    assert voiced.mean() > 0.7
    assert abs(np.median(f0[voiced]) - f0_true) / f0_true < 0.05


def test_silence_is_unvoiced():
    f0, voiced, _ = yin(np.zeros(SR, np.float32), SR)
    assert not voiced.any() and not f0.any()


def test_white_noise_is_unvoiced():
    x = np.random.default_rng(1).normal(0, 0.1, SR).astype(np.float32)
    _, voiced, aper = yin(x, SR)
    assert voiced.mean() < 0.15, f"{voiced.mean():.0%} of white noise called voiced"
    assert aper.mean() > 0.3


def test_no_octave_error_on_rich_harmonics():
    """The classic YIN failure: locking to a subharmonic (f0/2)."""
    for f0_true in [120, 200]:
        f0, voiced, _ = yin(_tone(f0_true, harmonics=10), SR)
        est = np.median(f0[voiced])
        assert abs(est - f0_true / 2) / f0_true > 0.1, f"octave-halved to {est:.0f}"
        assert abs(est - f0_true) / f0_true < 0.03


def test_frames_are_independent():
    """Frame-local by construction: a frame's f0 must not depend on its
    neighbours. This is what makes the tracker safe near pause_start."""
    x = np.concatenate([_tone(150, 0.5), _tone(250, 0.5)])
    f0_full, v_full, _ = yin(x, SR)
    f0_half, v_half, _ = yin(x[:len(x) // 2], SR)
    k = len(f0_half)
    assert np.allclose(f0_full[:k], f0_half, atol=1e-5), "later audio changed earlier frames"
    assert np.array_equal(v_full[:k], v_half)


def test_agrees_with_pyin_on_real_speech():
    """Agreement with a trusted implementation, pooled over frames.

    Deliberately NOT averaged per-file: an early version of this test did that
    and one near-silent 3 s excerpt (8 voiced frames, pyin railed at its own
    400 Hz ceiling and reporting a harmonic of noise) dominated the mean. We
    pool frames, demand a decent sample, and drop frames where pyin has railed
    -- comparing against a reference where the reference is broken measures
    nothing.
    """
    import librosa
    import pandas as pd
    mine, theirs, agree_n, agree_d = [], [], 0, 0
    for lang in ("english", "hindi"):
        df = pd.read_csv(f"{ROOT}/eot_handout/eot_data/{lang}/labels.csv")
        for _, r in df.head(8).iterrows():
            x, sr = sf.read(f"{ROOT}/eot_handout/eot_data/{lang}/{r.audio_file}", dtype="float32")
            seg = x[:int(r.pause_start * sr)][-int(3.0 * sr):]
            if len(seg) < sr:
                continue
            f0_m, v_m, _ = yin(seg, sr, frame_len=640, hop=160)
            f0_p, v_p, _ = librosa.pyin(seg, fmin=60, fmax=400, sr=sr, frame_length=640,
                                        hop_length=160, center=False)
            k = min(len(v_m), len(v_p))
            v_p = np.asarray(v_p[:k], bool)
            f0_p = np.nan_to_num(np.asarray(f0_p[:k]))
            agree_n += int((v_m[:k] == v_p).sum())
            agree_d += k
            railed = (f0_p > 390) | (f0_p < 62)          # pyin stuck on its own bounds
            both = v_m[:k] & v_p & ~railed
            mine.append(f0_m[:k][both])
            theirs.append(f0_p[both])
    mine, theirs = np.concatenate(mine), np.concatenate(theirs)
    assert len(mine) > 200, f"only {len(mine)} comparable frames -- test is not meaningful"
    voicing_agreement = agree_n / agree_d
    err = np.abs(mine - theirs) / theirs
    assert voicing_agreement > 0.80, f"voicing agreement with pyin only {voicing_agreement:.0%}"
    assert np.median(err) < 0.06, f"median f0 disagreement {np.median(err):.1%}"
    # a few octave errors are tolerable; systematic ones are not
    assert np.mean(err > 0.4) < 0.10, f"{np.mean(err>0.4):.0%} of frames are gross mismatches"


def test_voicing_is_not_degenerate_on_real_speech():
    """The bug this whole module exists to fix: yin+value-gating called 99% of
    frames voiced. Real conversational speech is roughly 40-90% voiced."""
    x, sr = sf.read(f"{ROOT}/eot_handout/eot_data/hindi/audio/hi__000.wav", dtype="float32")
    _, voiced, _ = yin(x[:int(15.7 * sr)], sr)
    assert 0.30 < voiced.mean() < 0.95, f"voiced fraction {voiced.mean():.0%} is implausible"


def test_is_fast_enough_to_ship():
    """predict.py must finish on a laptop. pyin cost ~4.7 s per pause."""
    import time
    x = _tone(150, 10.0)
    yin(x, SR)
    t = time.time()
    for _ in range(5):
        yin(x, SR)
    per_call = (time.time() - t) / 5
    assert per_call < 0.35, f"{per_call*1000:.0f} ms per 10 s window is too slow"


def test_voiced_probability_range():
    p = voiced_probability(np.array([0.0, 0.15, 0.3, 1.0]))
    assert p[0] == 1.0 and p[-1] == 0.0 and np.all((p >= 0) & (p <= 1))
