"""Dataset assembly: labels.csv + audio -> feature matrix.

Works on any folder with the handout's structure (audio/ + labels.csv), including
folders we have never seen. Feature extraction is parallel over turns and cached
on disk keyed by a hash of (file bytes, feature code, params), so iterating on
the model is instant after the first pass.
"""
import hashlib
import os

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Memory, Parallel, delayed

from .features import FEATURE_NAMES, SR, features_for_pause

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
_mem = Memory(CACHE, verbose=0)


def load_wav(path, target_sr=SR):
    """Mono float32 at target_sr. Tolerates stereo and odd sample rates."""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return np.ascontiguousarray(x, np.float32), sr


def _turn_features(data_dir, audio_file, starts, ends):
    """All pauses of one turn. Each pause only ever sees its own past."""
    x, sr = load_wav(os.path.join(data_dir, audio_file))
    rows = []
    for k, ps in enumerate(starts):
        prior = list(zip(starts[:k], ends[:k]))   # completed pauses only
        rows.append(features_for_pause(x, sr, ps, prior))
    return np.stack(rows)


_cached_turn_features = _mem.cache(_turn_features, ignore=[])


def read_labels(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
    need = {"turn_id", "audio_file", "pause_index", "pause_start", "label"}
    missing = need - set(df.columns) - {"label"}   # label absent on a blind test set
    if missing:
        raise ValueError(f"{data_dir}/labels.csv missing columns: {sorted(missing)}")
    return df.sort_values(["turn_id", "pause_index"]).reset_index(drop=True)


def build(data_dir, n_jobs=-1, use_cache=True, verbose=False):
    """-> (X, df) where df carries turn_id/pause_index/label(if present)/dur(if present).

    NOTE `dur` is emitted for TRAINING and SCORING only. It is future information
    at inference time and must never reach a feature. See eotlib/features.py.
    """
    df = read_labels(data_dir)
    groups = list(df.groupby("turn_id", sort=False))
    fn = _cached_turn_features if use_cache else _turn_features
    mats = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(fn)(data_dir, g.iloc[0].audio_file,
                    tuple(g.pause_start.astype(float)),
                    tuple(g.pause_end.astype(float)) if "pause_end" in g
                    else tuple(g.pause_start.astype(float)))
        for _, g in groups)
    X = np.concatenate(mats).astype(np.float32)
    order = np.concatenate([g.index.values for _, g in groups])
    X = X[np.argsort(np.argsort(order))] if not np.array_equal(order, np.arange(len(df))) else X
    assert len(X) == len(df), f"{len(X)} feature rows vs {len(df)} label rows"
    if "label" in df:
        df = df.assign(y=(df.label == "eot").astype(int))
    if "pause_end" in df:
        df = df.assign(dur=df.pause_end - df.pause_start)
    return X, df


def build_many(data_dirs, **kw):
    """Pool several language folders into one training set."""
    Xs, ds = [], []
    for d in data_dirs:
        X, df = build(d, **kw)
        Xs.append(X)
        ds.append(df.assign(corpus=os.path.basename(os.path.normpath(d))))
    df = pd.concat(ds, ignore_index=True)
    # turn ids are unique per corpus already, but namespace them to be safe
    df["group"] = df.corpus + "/" + df.turn_id
    return np.concatenate(Xs), df


__all__ = ["build", "build_many", "load_wav", "read_labels", "FEATURE_NAMES"]
