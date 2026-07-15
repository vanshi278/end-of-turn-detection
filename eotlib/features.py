"""Causal prosodic features for end-of-turn detection.

CAUSALITY CONTRACT
------------------
`features_for_pause` is the ONLY public entry point. Its first act is to
truncate the waveform at pause_start and hand the truncated copy to `_features`,
which never receives the full signal, `pause_start`, or `pause_end`. Feature
code therefore *cannot* look at the future -- it is not merely discouraged from
doing so. tests/test_causality.py proves it by appending arbitrary audio after
each pause and asserting the features are bit-identical.

Traps this file deliberately avoids (each one silently scores ~AUC 1.0 on this
dataset because every WAV ends exactly at its eot pause):
  * len(x)/sr, or any "time remaining in file" quantity
  * pause_end / pause_duration of the CURRENT pause (future information)
  * the number of rows this turn has in labels.csv (reveals which pause is last)
  * pitch trackers with global smoothing (e.g. librosa.pyin's Viterbi decode)
    run over the whole file and then sliced -- frames near pause_start would be
    informed by frames after it. We truncate FIRST, then track, always.

Prior pauses are legal: they are complete and in the past. The current pause's
duration is not -- a live agent is sitting inside that silence, not looking back
at it.
"""
import numpy as np
import librosa

from .yin import yin as _yin, voiced_probability as _voiced_probability

SR = 16000
# One 40 ms / 10 ms grid for pitch AND energy, so frame i means the same samples
# in both. Mixed frame sizes made the "energy at the final voiced run" features
# quietly off-by-a-few-frames.
FRAME_MS, HOP_MS = 40, 10
TAIL_S = 1.5      # the "last breath" window the prosody lives in
HIST_S = 10.0     # bounded speaker-normalisation history (caps compute)
F0_MIN, F0_MAX = 60.0, 400.0

FEATURE_NAMES = [
    # --- WHERE IN THE SPEAKER'S OWN RANGE the turn lands.
    # From the human listening study (72% blind, 87.5% on confident Hindi calls):
    # the cue driving nearly every correct call was "pitch dropped / low pitch
    # ending" -> eot, and "pitch was very high, cant be the end" -> hold. Note
    # "high" and "low" are relative to THAT speaker -- so the right encoding is a
    # rank inside their own pitch distribution so far, not a z-score. A z-score
    # asks "how many IQRs from your median", which is not what the ear is doing;
    # the ear asks "did you come home to your floor, or stop up in your range?"
    "f0_final_pct", "f0_min_pct", "f0_end_vs_floor", "f0_end_vs_ceil",
    # --- final pitch movement (statements fall; continuations stay level/rise)
    "f0_slope_final", "f0_slope_tail", "f0_final_z", "f0_delta_final",
    "f0_range_tail_z", "f0_last_minus_min", "f0_slope_150ms", "f0_reset_z",
    # --- pitch FLATNESS: a held monotone vowel is a hesitation ("uhh", "matlab"),
    #     i.e. the speaker is buying time -> hold. A turn end moves the pitch.
    "f0_std_final", "f0_flat_score",
    # --- final lengthening (last syllable stretches at a turn end)
    "voiced_run_final_s", "voiced_run_ratio", "voiced_frac_tail",
    "unvoiced_frac_final",
    # --- energy decay into the pause
    "e_slope_final", "e_slope_tail", "e_final_z", "e_ratio_200_1000", "e_drop_300",
    # --- voice quality (creak/breathiness at turn ends)
    "centroid_final_z", "tilt_final", "zcr_final_z", "flux_final",
    "flatness_final", "vprob_final", "vprob_slope",
    # --- BREATH / trailing aspiration. Human study: "person taking breath" ->
    # hold (correct, conf 4). An audible inhalation means the speaker is
    # reloading, not finishing. It shows up as a trailing unvoiced, noisy,
    # low-energy stretch with a high, flat spectrum.
    "trail_unvoiced_s", "trail_centroid_z", "trail_flatness", "trail_energy_z",
    # --- ABRUPTNESS. Human study: "abrupt stopping" -> hold. A turn end decays;
    # a hesitation gets guillotined mid-word.
    "abruptness",
    # --- VOWEL IDENTITY of the final nucleus, via cepstral-mean-normalised
    # MFCCs. The human read Hindi verb endings lexically ("rahega used",
    # "ended with rakhiyega", "baat karke gaya tha") -- we cannot do ASR, but
    # Hindi is verb-final and its finite verbs land on a small set of vowel
    # qualities (-ega/-yega/-tha/-hai/-iye). MFCCs encode vowel quality; CMN
    # against this speaker's own history so far removes channel and speaker.
    "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5", "mfcc6", "mfcc7", "mfcc8",
    # --- rhythm / speaking rate (speakers slow down before finishing)
    "rate_tail", "rate_ratio",
    # --- LOCAL final slowing. From the error study: "speaking fast, end mai
    # usually we speak slowly". rate_ratio compares a 1.5 s tail against a 10 s
    # history, which is far too blunt to see a speaker decelerating INTO the
    # pause; these compare the last ~500 ms against the second before it.
    # Strong in English (0.694 / 0.669), weak in Hindi -- right cue, and it
    # turns out the languages differ on it.
    "rate_final_500", "rate_decel",
    # --- PROMINENCE of the last syllable. From the error study: "aur is not
    # end, it is conjunction". Hindi function words that promise more (aur, toh,
    # ki) are short and unstressed; a turn-final predicate carries stress. So
    # compare the final voiced run against the one before it in both energy and
    # duration -- a weak, short final syllable means a function word is dangling.
    "prom_energy_final", "prom_dur_final",
    # Two more round-2 candidates were built and CUT for carrying no signal:
    # "rhythm_regularity" (list intonation, from "number is not the end") scored
    # 0.512, and a local final-lengthening ratio scored 0.516. The perception is
    # real; energy-peak timing is evidently not how to measure it.
    # --- causal turn context
    "elapsed_s", "n_prior_pauses", "prior_pause_mean_s", "prior_pause_max_s",
    "speech_run_s", "voiced_frac_hist",
]
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------- utilities
def _frame_energy_db(x, sr):
    fl, hp = int(sr * FRAME_MS / 1000), int(sr * HOP_MS / 1000)
    if len(x) < fl:
        return np.zeros(0, np.float32)
    rms = librosa.feature.rms(y=x, frame_length=fl, hop_length=hp, center=False)[0]
    return (20 * np.log10(rms + 1e-12)).astype(np.float32)


def _f0(x, sr):
    """Pitch + voicing from our own vectorised YIN (see eotlib/yin.py).

    Validated against librosa.pyin: 85% voicing agreement, 0.42% median f0
    disagreement over 1832 real speech frames, and 68x faster (40 ms vs 2.7 s
    per 10 s window). The speed is what makes predict.py shippable; the voicing
    is what makes final-lengthening features mean anything.

    Every frame depends only on its own samples, so nothing can flow backwards
    across pause_start.

    -> (f0 Hz with 0 where unvoiced, voiced_probability in [0,1])
    """
    fl = int(sr * FRAME_MS / 1000)
    if len(x) < fl * 2:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    f0, voiced, aper = _yin(x, sr, fmin=F0_MIN, fmax=F0_MAX, frame_len=fl,
                            hop=int(sr * HOP_MS / 1000))
    return f0, _voiced_probability(aper)


def _slope(v, dt=HOP_MS / 1000.0):
    """Least-squares slope in units/sec. 0 when under-determined."""
    if len(v) < 2:
        return 0.0
    t = np.arange(len(v), dtype=np.float64) * dt
    t -= t.mean()
    den = (t * t).sum()
    return float((t * (v - v.mean())).sum() / den) if den > 0 else 0.0


def _runs(mask):
    """[(start, stop)) index pairs of True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _z(v, ref):
    """Robust z-score of v against a reference distribution (speaker-relative).

    This is what makes the model transfer across speakers, channels and
    languages: absolute Hz is a property of the person, the *departure* from
    their own baseline is the turn-ending cue.
    """
    if len(ref) < 3:
        return 0.0
    med = np.median(ref)
    iqr = np.subtract(*np.percentile(ref, [75, 25]))
    return float((v - med) / (iqr + 1e-6))


def _pct(v, ref):
    """Rank of v inside this speaker's own distribution so far, in [0,1].

    The listening study's dominant cue was "did the pitch land LOW or HIGH *for
    this person*". A z-score answers "how many IQRs from your median" and is
    symmetric and unbounded; a rank answers the question the ear is actually
    asking, is bounded, and is immune to the skew of a pitch distribution.
    Returns 0.5 (uninformative) when there is not enough history to rank against.
    """
    if len(ref) < 5:
        return 0.5
    return float((np.asarray(ref) < v).mean())


def _peaks(e_db):
    """Syllable-nucleus-ish peaks in an energy contour."""
    if len(e_db) < 5:
        return np.zeros(0, int)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(e_db - e_db.mean(), distance=8, prominence=3.0)
    return pk


def _rate(e_db):
    """Syllable-ish rate: energy-envelope peaks per second."""
    if len(e_db) < 5:
        return 0.0
    return float(len(_peaks(e_db)) / (len(e_db) * HOP_MS / 1000.0 + 1e-6))


# ------------------------------------------------------------------ public
def features_for_pause(x, sr, pause_start, prior_pauses=()):
    """Features for the pause beginning at `pause_start`.

    x            full waveform (mono, float32)
    pause_start  seconds
    prior_pauses iterable of (start, end) for pauses that COMPLETED before
                 pause_start. Past silences are legal; this one is not.

    Everything after pause_start is discarded here, on line one, and is never
    passed on.
    """
    n = int(round(pause_start * sr))
    xs = np.asarray(x[:n], dtype=np.float32)
    prior = [(s, e) for (s, e) in prior_pauses if e <= pause_start + 1e-9]
    return _features(xs, sr, prior)


def _features(xs, sr, prior):
    """The future does not exist in this function. It only has `xs`."""
    out = np.zeros(N_FEATURES, np.float32)
    if len(xs) < int(0.2 * sr):
        return out

    hist = xs[-int(HIST_S * sr):] if len(xs) > int(HIST_S * sr) else xs

    # ONE pitch pass over the history. The tail is a SUFFIX of the history, so
    # tail statistics are a slice -- not a second (expensive) pyin call.
    e_h = _frame_energy_db(hist, sr)
    f0_h, vp_h = _f0(hist, sr)
    nf_ = min(len(e_h), len(f0_h))
    if nf_ < 3:
        return out
    e_h, f0_h, vp_h = e_h[:nf_], f0_h[:nf_], vp_h[:nf_]

    n_tail = min(nf_, int(TAIL_S * 1000 / HOP_MS))
    e_t, f0_t, vp_t = e_h[-n_tail:], f0_h[-n_tail:], vp_h[-n_tail:]
    tail = hist[-int(TAIL_S * sr):] if len(hist) > int(TAIL_S * sr) else hist
    vh, vt = f0_h[f0_h > 0], f0_t[f0_t > 0]

    # ---- final voiced run: the last syllable, where the turn-final cue lives
    vmask = f0_t > 0
    runs = _runs(vmask)
    fin = runs[-1] if runs else None
    f0_fin = f0_t[fin[0]:fin[1]] if fin else np.zeros(0, np.float32)

    i = {n_: k for k, n_ in enumerate(FEATURE_NAMES)}

    # ---- where in the speaker's own range did they stop? (the human's cue)
    out[i["f0_final_pct"]] = 0.5
    out[i["f0_min_pct"]] = 0.5
    if len(f0_fin) and len(vh) >= 5:
        f0_end = float(np.median(f0_fin[-5:]))          # pitch at the very end
        out[i["f0_final_pct"]] = _pct(f0_end, vh)
        out[i["f0_min_pct"]] = _pct(float(f0_fin.min()), vh)
        floor, ceil = np.percentile(vh, 5), np.percentile(vh, 95)
        rng = max(ceil - floor, 1e-6)
        out[i["f0_end_vs_floor"]] = float((f0_end - floor) / rng)
        out[i["f0_end_vs_ceil"]] = float((ceil - f0_end) / rng)

    if len(f0_fin) >= 2:
        out[i["f0_slope_final"]] = _slope(f0_fin)
        out[i["f0_delta_final"]] = float(f0_fin[-1] - f0_fin[0])
        out[i["f0_last_minus_min"]] = float(f0_fin[-1] - f0_fin.min())
        out[i["f0_std_final"]] = float(f0_fin.std())
        # long AND monotone == a sustained filler vowel ("uhh", "matlab...")
        # i.e. the speaker is buying time. A real ending moves the pitch.
        out[i["f0_flat_score"]] = (len(f0_fin) * HOP_MS / 1000.0) / (float(f0_fin.std()) + 1.0)
        out[i["f0_reset_z"]] = _z(float(f0_fin[0]), vh)
    if len(vt) >= 2:
        out[i["f0_slope_tail"]] = _slope(vt)
        out[i["f0_slope_150ms"]] = _slope(vt[-15:])
    if len(f0_fin):
        out[i["f0_final_z"]] = _z(float(np.median(f0_fin)), vh)
    if len(vt) >= 2:
        out[i["f0_range_tail_z"]] = _z(float(vt.max() - vt.min()), vh)

    # ---- final lengthening: last syllable vs this speaker's own average
    if fin:
        fl_s = (fin[1] - fin[0]) * HOP_MS / 1000.0
        out[i["voiced_run_final_s"]] = fl_s
        hruns = [(b - a) * HOP_MS / 1000.0 for a, b in _runs(f0_h > 0)]
        if len(hruns) >= 2:
            out[i["voiced_run_ratio"]] = fl_s / (np.mean(hruns) + 1e-6)
    out[i["voiced_frac_tail"]] = float(vmask.mean()) if len(vmask) else 0.0
    out[i["unvoiced_frac_final"]] = float(1.0 - vmask[-50:].mean()) if len(vmask) else 0.0

    # ---- energy decay into the silence
    if len(e_t) >= 2:
        out[i["e_slope_tail"]] = _slope(e_t)
        out[i["e_slope_final"]] = _slope(e_t[-30:])          # last 300 ms
        out[i["e_final_z"]] = _z(float(e_t[-5:].mean()), e_h)
        n200, n1k = min(20, len(e_t)), min(100, len(e_t))
        out[i["e_ratio_200_1000"]] = float(e_t[-n200:].mean() - e_t[-n1k:].mean())
        out[i["e_drop_300"]] = float(e_t[-30:].max() - e_t[-3:].mean()) if len(e_t) >= 30 else 0.0

    # ---- voice quality on the final voiced stretch
    if fin:
        out[i["vprob_final"]] = float(vp_t[fin[0]:fin[1]].mean())
        out[i["vprob_slope"]] = _slope(vp_t[-30:]) if len(vp_t) >= 5 else 0.0
        hop_n = int(sr * HOP_MS / 1000)
        a = max(0, fin[0] * hop_n)
        b = min(len(tail), fin[1] * hop_n + int(sr * FRAME_MS / 1000))
        seg = tail[a:b]
        if len(seg) >= int(0.03 * sr):
            # n_fft must fit the segment: librosa's 2048 default zero-pads short
            # final syllables, which biases the centroid low exactly where we
            # are trying to measure creak.
            nf = int(2 ** np.floor(np.log2(min(len(seg), 1024))))
            cen = librosa.feature.spectral_centroid(y=seg, sr=sr, n_fft=nf, hop_length=nf // 2)[0]
            cen_h = librosa.feature.spectral_centroid(y=hist, sr=sr, n_fft=1024, hop_length=512)[0]
            out[i["centroid_final_z"]] = _z(float(cen.mean()), cen_h)
            S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            fr = np.fft.rfftfreq(len(seg), 1 / sr)
            lo, hi = S[(fr > 80) & (fr < 1000)].mean(), S[(fr > 2000) & (fr < 6000)].mean()
            out[i["tilt_final"]] = float(20 * np.log10((lo + 1e-9) / (hi + 1e-9)))
            z = librosa.feature.zero_crossing_rate(seg, frame_length=512, hop_length=160)[0]
            z_h = librosa.feature.zero_crossing_rate(hist, frame_length=512, hop_length=160)[0]
            out[i["zcr_final_z"]] = _z(float(z.mean()), z_h)
            # low spectral variation across a long voiced run == sustained vowel
            out[i["flatness_final"]] = float(np.mean(
                librosa.feature.spectral_flatness(y=seg, n_fft=nf, hop_length=nf // 2)[0]))
            if len(seg) > int(0.06 * sr):
                m = np.abs(librosa.stft(seg, n_fft=512, hop_length=160, center=False))
                out[i["flux_final"]] = float(np.mean(np.abs(np.diff(m, axis=1)))) if m.shape[1] > 1 else 0.0

    # ---- breath / trailing aspiration: "person taking breath" -> hold
    # The trailing unvoiced run right before the silence. An inhalation is
    # unvoiced, noisy (flat spectrum), high-centroid and quiet.
    hop_n = int(sr * HOP_MS / 1000)
    uv_runs = _runs(~vmask)
    if uv_runs and uv_runs[-1][1] == len(vmask):        # unvoiced AT the offset
        a_, b_ = uv_runs[-1]
        out[i["trail_unvoiced_s"]] = (b_ - a_) * HOP_MS / 1000.0
        out[i["trail_energy_z"]] = _z(float(e_t[a_:b_].mean()), e_h)
        seg_u = tail[a_ * hop_n: min(len(tail), b_ * hop_n + int(sr * FRAME_MS / 1000))]
        if len(seg_u) >= int(0.02 * sr):
            nfu = int(2 ** np.floor(np.log2(min(len(seg_u), 1024))))
            if nfu >= 32:
                cu = librosa.feature.spectral_centroid(y=seg_u, sr=sr, n_fft=nfu,
                                                       hop_length=nfu // 2)[0]
                ch = librosa.feature.spectral_centroid(y=hist, sr=sr, n_fft=1024,
                                                       hop_length=512)[0]
                out[i["trail_centroid_z"]] = _z(float(cu.mean()), ch)
                out[i["trail_flatness"]] = float(np.mean(
                    librosa.feature.spectral_flatness(y=seg_u, n_fft=nfu,
                                                      hop_length=nfu // 2)[0]))

    # ---- abruptness: "abrupt stopping" -> hold. A turn end rolls off; a
    # hesitation is cut mid-word, so the last frame is still near full energy.
    if len(e_t) >= 12:
        out[i["abruptness"]] = float(e_t[-1] - e_t[-12:-2].min())

    # ---- vowel identity of the final nucleus, CMN'd against this speaker
    if fin:
        a_ = max(0, fin[0] * hop_n)
        b_ = min(len(tail), fin[1] * hop_n + int(sr * FRAME_MS / 1000))
        nucleus = tail[a_:b_]
        if len(nucleus) >= int(0.04 * sr) and len(hist) >= int(0.5 * sr):
            nfm = int(2 ** np.floor(np.log2(min(len(nucleus), 1024))))
            if nfm >= 128:
                m_fin = librosa.feature.mfcc(y=nucleus, sr=sr, n_mfcc=9, n_fft=nfm,
                                             hop_length=nfm // 2).mean(axis=1)
                m_hist = librosa.feature.mfcc(y=hist, sr=sr, n_mfcc=9, n_fft=1024,
                                              hop_length=512).mean(axis=1)
                cmn = (m_fin - m_hist)[1:9]      # drop c0 (loudness), keep shape
                for k_ in range(8):
                    out[i[f"mfcc{k_+1}"]] = float(cmn[k_])

    # ---- rhythm: slowing down is a turn-ending cue
    r_t, r_h = _rate(e_t), _rate(e_h)
    out[i["rate_tail"]] = r_t
    out[i["rate_ratio"]] = float(r_t / (r_h + 1e-6))

    # ---- LOCAL deceleration into the pause ("end mai usually we speak slowly")
    if len(e_t) >= 60:
        last500, prev1000 = e_t[-50:], e_t[-150:-50]
        rf = _rate(last500)
        out[i["rate_final_500"]] = rf
        if len(prev1000) >= 20:
            out[i["rate_decel"]] = float(rf / (_rate(prev1000) + 1e-6))

    # ---- prominence of the final syllable vs the one before it. A dangling
    # unstressed function word (aur / toh / ki) is quiet and short.
    if runs and len(runs) >= 2:
        (a1, b1), (a0, b0) = runs[-1], runs[-2]
        e_fin, e_prev = e_t[a1:b1], e_t[a0:b0]
        if len(e_fin) and len(e_prev):
            out[i["prom_energy_final"]] = float(e_fin.max() - e_prev.max())
            out[i["prom_dur_final"]] = float((b1 - a1) / max(b0 - a0, 1))


    # ---- causal turn context (no future, no file length)
    out[i["elapsed_s"]] = len(xs) / sr
    out[i["n_prior_pauses"]] = len(prior)
    if prior:
        d = [e - s for s, e in prior]
        out[i["prior_pause_mean_s"]] = float(np.mean(d))
        out[i["prior_pause_max_s"]] = float(np.max(d))
        out[i["speech_run_s"]] = len(xs) / sr - prior[-1][1]   # since last resume
    else:
        out[i["speech_run_s"]] = len(xs) / sr
    out[i["voiced_frac_hist"]] = float((f0_h > 0).mean()) if len(f0_h) else 0.0

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
