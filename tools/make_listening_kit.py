"""Build a BLIND listening kit for human error analysis.

For each sampled pause we export two clips:
  <id>__context.wav  audio ending exactly at pause_start -- EXACTLY what the
                     model is allowed to hear. Listen to this, then guess.
  <id>__reveal.wav   the same context plus what actually happened next.
                     Open only after you have written your guess down.

Sampling is metric-aware: a `hold` pause only costs us if its duration exceeds
the agent's action delay, so we over-sample LONG holds (the ones that actually
cause false cutoffs) and ignore the short ones that are free.

    python tools/make_listening_kit.py --n_hindi 24 --n_english 12
"""
import argparse
import os
import numpy as np
import pandas as pd
import soundfile as sf

DATA = "/Users/vanshikaagarwal/speedrun/eot_handout/eot_data"
OUT = "/Users/vanshikaagarwal/speedrun/listening_kit"
CONTEXT_S = 3.0          # how much speech before the pause you hear
REVEAL_AFTER_S = 2.5     # how much of the future the reveal clip exposes
EXPENSIVE_HOLD_S = 0.55  # holds at least this long are the ones that cost us


def sample_pauses(lang, n, rng):
    df = pd.read_csv(os.path.join(DATA, lang, "labels.csv"))
    df["dur"] = df.pause_end - df.pause_start
    n_eot = n // 2
    eots = df[df.label == "eot"].sample(n_eot, random_state=rng)
    # only the holds that can actually trigger a false cutoff
    holds = df[(df.label == "hold") & (df.dur >= EXPENSIVE_HOLD_S)]
    holds = holds.sample(min(n - n_eot, len(holds)), random_state=rng)
    return pd.concat([eots, holds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_hindi", type=int, default=24)
    ap.add_argument("--n_english", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = []
    for lang, n in [("hindi", args.n_hindi), ("english", args.n_english)]:
        rows.append(sample_pauses(lang, n, args.seed).assign(lang=lang))
    picks = pd.concat(rows).sample(frac=1.0, random_state=args.seed)  # shuffle => blind

    os.makedirs(OUT, exist_ok=True)
    key = []
    for i, (_, r) in enumerate(picks.iterrows(), start=1):
        cid = f"clip_{i:03d}"
        x, sr = sf.read(os.path.join(DATA, r.lang, r.audio_file), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        ps = int(r.pause_start * sr)
        lo = max(0, ps - int(CONTEXT_S * sr))
        hi = min(len(x), int((r.pause_end + REVEAL_AFTER_S) * sr))
        sf.write(os.path.join(OUT, f"{cid}__context.wav"), x[lo:ps], sr)
        sf.write(os.path.join(OUT, f"{cid}__reveal.wav"), x[lo:hi], sr)
        key.append({"clip_id": cid, "lang": r.lang, "turn_id": r.turn_id,
                    "pause_index": r.pause_index, "truth": r.label,
                    "pause_dur": round(float(r.dur), 3)})

    key = pd.DataFrame(key)
    key.to_csv(os.path.join(OUT, "ANSWER_KEY.csv"), index=False)   # do not peek
    # the sheet you fill in
    sheet = key[["clip_id"]].copy()
    sheet["your_guess"] = ""        # hold | eot
    sheet["confidence"] = ""        # 1 (coin flip) .. 5 (certain)
    sheet["what_you_heard"] = ""    # free text -- THIS is where the features come from
    sheet.to_csv(os.path.join(OUT, "MY_GUESSES.csv"), index=False)

    print(f"wrote {len(key)} clip pairs -> {OUT}")
    print(f"  hindi={sum(key.lang=='hindi')}  english={sum(key.lang=='english')}")
    print(f"  truth balance: {key.truth.value_counts().to_dict()}")
    print("fill in listening_kit/MY_GUESSES.csv ; ANSWER_KEY.csv stays closed until you are done")


if __name__ == "__main__":
    main()
