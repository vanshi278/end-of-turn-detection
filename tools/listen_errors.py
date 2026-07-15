"""Human error analysis on the model's WORST, most expensive mistakes.

    python tools/listen_errors.py --build    # cut the clips
    python tools/listen_errors.py            # listen + annotate

Not blind. You are told the truth up front and asked WHY -- the goal is to
elicit the cue the model is missing, not to test you again.

Which errors we pick is metric-aware. Only two kinds of mistake cost anything:

  FALSE ALARM  a hold LONGER than the agent's delay that we score high. This is
               what interrupts a real user, and only ~25 pauses in Hindi can do
               it at all. Short holds are free and are ignored here.
  MISS         a true turn end we score low, so the agent sits through the full
               1.6 s timeout.

Predictions are out-of-fold: every pause is scored by a model that never saw its
turn, so these are real generalisation errors, not memorisation artefacts.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, "/Users/vanshikaagarwal/speedrun")

KIT = "/Users/vanshikaagarwal/speedrun/error_kit"
DATA = "/Users/vanshikaagarwal/speedrun/eot_handout/eot_data"
CONTEXT_S = 3.0
REVEAL_AFTER_S = 2.5
COST_D = 0.6      # holds longer than this are the ones that can interrupt


def build(n_per_type=8, langs=("hindi", "english")):
    from eotlib.cv import oof_predict
    from eotlib.data import build_many
    from train import build_model, cost_weights, DEFAULT_DIRS

    X, df = build_many(DEFAULT_DIRS)
    y, dur = df.y.values, df.dur.values
    p = np.mean([oof_predict(build_model(), X, y, df.group.values, seed=s,
                             sample_weight=cost_weights(y, dur)) for s in (0, 1, 2)], axis=0)
    df = df.assign(p=p)

    os.makedirs(KIT, exist_ok=True)
    rows = []
    for lang in langs:
        d = df[df.corpus == lang]
        # false alarms: expensive holds we were most confident about
        fa = d[(d.y == 0) & (d.dur > COST_D)].nlargest(n_per_type, "p")
        # misses: true turn ends we were least confident about
        ms = d[d.y == 1].nsmallest(n_per_type, "p")
        for kind, sub in [("FALSE_ALARM", fa), ("MISS", ms)]:
            for _, r in sub.iterrows():
                rows.append({"kind": kind, "lang": lang, "turn_id": r.turn_id,
                             "pause_index": int(r.pause_index), "p_eot": round(float(r.p), 3),
                             "truth": r.label, "pause_dur": round(float(r.dur), 2),
                             "audio_file": r.audio_file, "pause_start": float(r.pause_start),
                             "pause_end": float(r.pause_end)})
    man = pd.DataFrame(rows)
    # hindi first: the hidden set is mostly Hindi
    man["_o"] = (man.lang != "hindi").astype(int)
    man = man.sort_values(["_o", "kind"]).drop(columns="_o").reset_index(drop=True)
    man.insert(0, "clip_id", [f"err_{i:02d}" for i in range(len(man))])

    for _, r in man.iterrows():
        x, sr = sf.read(os.path.join(DATA, r.lang, r.audio_file), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        ps = int(r.pause_start * sr)
        lo = max(0, ps - int(CONTEXT_S * sr))
        hi_ = min(len(x), int((r.pause_end + REVEAL_AFTER_S) * sr))
        ctx, rev = x[lo:ps], x[lo:hi_]
        # Normalise for the ear only. Phone audio sits around RMS 0.05, which is
        # tiring to judge 32 times over; peak-normalising makes the cue audible
        # without changing anything the MODEL sees -- these clips are a listening
        # aid, the feature extractor reads the original WAVs.
        # Gain is computed on the CONTEXT and applied to both, so the reveal does
        # not jump in level between the two clips of the same pause.
        g = 0.95 / max(float(np.abs(ctx).max()), 1e-6)
        sf.write(os.path.join(KIT, f"{r.clip_id}__context.wav"),
                 np.clip(ctx * g, -1, 1), sr)
        sf.write(os.path.join(KIT, f"{r.clip_id}__reveal.wav"),
                 np.clip(rev * g, -1, 1), sr)
    man["why"] = ""
    man.to_csv(os.path.join(KIT, "ERRORS.csv"), index=False)
    print(f"wrote {len(man)} error clips -> {KIT}")
    print(man.groupby(["lang", "kind"]).size().to_string())


def play(p, volume=1.0):
    """afplay -v boosts beyond unity; the clips are already peak-normalised."""
    r = subprocess.run(["afplay", "-v", str(volume), p],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [audio failed: {r.stderr.strip() or 'afplay exit ' + str(r.returncode)}]")
        print(f"  try in another terminal:  afplay {p}")


def selftest():
    """Confirm audio actually reaches your ears before spending 20 minutes."""
    import numpy as _np
    tone = (0.3 * _np.sin(2 * _np.pi * 440 * _np.arange(int(0.6 * 16000)) / 16000))
    t = os.path.join(KIT, "_selftest.wav")
    sf.write(t, tone.astype("float32"), 16000)
    print("1/2  playing a 440 Hz beep ...")
    play(t)
    os.remove(t)
    first = sorted(f for f in os.listdir(KIT) if f.endswith("__context.wav"))
    if first:
        print(f"2/2  playing a real clip ({first[0]}, 3 s of speech) ...")
        play(os.path.join(KIT, first[0]))
    print("\nHeard both?  -> run: python tools/listen_errors.py")
    print("Heard neither -> macOS output device (Control Centre > Sound), not the script.")
    print("Beep only     -> tell me; the clip export is at fault.")


def session():
    m = pd.read_csv(os.path.join(KIT, "ERRORS.csv"), dtype={"why": str}).fillna({"why": ""})
    todo = m[m.why == ""]
    print(f"\n{len(todo)} of {len(m)} left. The model is WRONG on every one of these.")
    print("You hear only what the model heard (audio ends exactly at the pause).")
    print("  FALSE_ALARM = really a HOLD, we thought turn was over -> we'd cut the user off")
    print("  MISS        = really the END, we thought they'd continue -> user waits 1.6s")
    print("Commands: r replay | v play reveal (hear what happened next) | s skip | q quit\n")
    for i in todo.index:
        r = m.loc[i]
        ctx = os.path.join(KIT, f"{r.clip_id}__context.wav")
        print(f"\n--- {r.clip_id}  [{r.lang}]  {r.kind}   truth={r.truth}  "
              f"model said p_eot={r.p_eot}")
        input("    [enter to play] ")     # play on YOUR cue, not before you look up
        play(ctx)
        while True:
            a = input("  why is this really " + r.truth.upper() + "? (r/v/s/q or type): ").strip()
            if a == "r":
                play(ctx); continue
            if a == "v":
                play(os.path.join(KIT, f"{r.clip_id}__reveal.wav")); continue
            if a == "s":
                break
            if a == "q":
                m.to_csv(os.path.join(KIT, "ERRORS.csv"), index=False)
                print("saved. rerun to resume."); return
            if a:
                m.at[i, "why"] = a
                m.to_csv(os.path.join(KIT, "ERRORS.csv"), index=False)
                break
    print(f"\nDone -> {KIT}/ERRORS.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="check your audio works before spending 20 minutes")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.selftest:
        selftest()
    else:
        session()
