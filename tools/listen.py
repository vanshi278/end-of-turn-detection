"""Interactive blind listening session. Run it in your own terminal:

    python tools/listen.py            # start / resume
    python tools/listen.py --review   # after finishing: score yourself + hear misses

Per clip you hear ONLY the audio the model is allowed to hear (ends exactly at
pause_start). Guess hold/eot, rate confidence, and -- most important -- type what
you actually heard. Progress saves after every clip, so you can stop anytime.
"""
import argparse
import os
import subprocess
import pandas as pd

KIT = "/Users/vanshikaagarwal/speedrun/listening_kit"
HELP = """
  h = hold  (speaker is going to keep talking)
  e = eot   (speaker is finished)
  r = replay      s = skip
  q = save & quit
"""


def play(path):
    subprocess.run(["afplay", path], check=False)


def load():
    g = pd.read_csv(os.path.join(KIT, "MY_GUESSES.csv"), dtype=str).fillna("")
    return g


def session():
    g = load()
    todo = g[g.your_guess == ""]
    print(f"\n{len(todo)} clips left of {len(g)}.")
    print("You hear speech that stops dead. Is the speaker DONE, or pausing mid-thought?")
    print(HELP)
    for i in todo.index:
        cid = g.at[i, "clip_id"]
        path = os.path.join(KIT, f"{cid}__context.wav")
        print(f"\n--- {cid}  ({list(g.index).index(i)+1}/{len(g)}) ---")
        play(path)
        while True:
            a = input("hold/eot [h/e/r/s/q]: ").strip().lower()
            if a == "r":
                play(path); continue
            if a == "q":
                g.to_csv(os.path.join(KIT, "MY_GUESSES.csv"), index=False)
                print("saved. rerun to resume."); return
            if a == "s":
                break
            if a in ("h", "e"):
                g.at[i, "your_guess"] = "hold" if a == "h" else "eot"
                c = input("confidence 1-5: ").strip()
                g.at[i, "confidence"] = c if c in list("12345") else "3"
                g.at[i, "what_you_heard"] = input("what made you say that? ").strip()
                g.to_csv(os.path.join(KIT, "MY_GUESSES.csv"), index=False)
                break
            print(HELP)
    print("\nAll done. Now run:  python tools/listen.py --review")


def review():
    g = load()
    k = pd.read_csv(os.path.join(KIT, "ANSWER_KEY.csv"))
    m = g.merge(k, on="clip_id")
    m = m[m.your_guess != ""]
    if not len(m):
        print("no guesses yet"); return
    m["correct"] = m.your_guess == m.truth
    print(f"\n=== YOUR BLIND ACCURACY: {m.correct.mean():.1%}  (n={len(m)}) ===")
    print(f"    chance = 50% (kit is balanced)\n")
    for lang in m.lang.unique():
        s = m[m.lang == lang]
        print(f"  {lang:8s} {s.correct.mean():.1%}  (n={len(s)})")
    print("\n  by confidence:")
    for c in sorted(m.confidence.unique()):
        s = m[m.confidence == c]
        print(f"    conf {c}: {s.correct.mean():.1%}  (n={len(s)})")
    print("\n  confusion:")
    print(pd.crosstab(m.truth, m.your_guess).to_string())
    m.to_csv(os.path.join(KIT, "RESULTS.csv"), index=False)

    wrong = m[~m.correct]
    print(f"\n=== {len(wrong)} MISSES -- listen to the reveal to hear what you missed ===")
    for _, r in wrong.iterrows():
        print(f"\n{r.clip_id}  truth={r.truth}  you={r.your_guess}  conf={r.confidence}")
        print(f"  you heard: {r.what_you_heard}")
        if input("  play reveal? [y/N] ").strip().lower() == "y":
            play(os.path.join(KIT, f"{r.clip_id}__reveal.wav"))
            note = input("  what did you actually miss? ").strip()
            m.loc[m.clip_id == r.clip_id, "post_hoc_note"] = note
            m.to_csv(os.path.join(KIT, "RESULTS.csv"), index=False)
    print(f"\nsaved -> {KIT}/RESULTS.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    review() if a.review else session()
