# NOTES

**What signal the model uses.** For each pause it reads only audio before
`pause_start` and asks four questions: where did the pitch land inside *this
speaker's own* range so far (a percentile, not a z-score — a turn end comes home
to the speaker's floor, a hesitation stops up in their range); did the final
voiced run trail into breath or aspiration (a speaker reloading, not finishing);
did the voice decay or get guillotined mid-word; and what vowel quality did the
final syllable carry, via cepstral-mean-normalised MFCCs — which is how Hindi's
verb-final endings (*-ega, -yega, -tha, -hai*) reach the model without ASR. These
run on our own vectorised YIN tracker (`eotlib/yin.py`), written because librosa's
`yin` offers no voicing decision and `pyin` costs 4.7 s/pause; ours agrees with
pyin to 0.42% on pitch at 68× the speed. Training weights each hold by whether it
could actually interrupt anyone — a hold shorter than the agent's delay is free,
so at a 0.6 s delay ~75% of holds are worth nothing and shouldn't consume model
capacity.

**Where it still fails.** Hindi: out-of-fold we match the 850 ms silence baseline
and the paired bootstrap puts our mean at 836 ms vs 882 ms with a 95% interval of
[0, 191] ms — the discrimination is real (hard_auc 0.71 vs 0.51 before the human
listening study) but it is not yet a demonstrable win in milliseconds, and we
report it that way rather than quoting the 511 ms in-sample number that the same
model produces on the folders it trained on. The residual errors are the ones our
human listener also got wrong: syntactically complete clauses where the speaker
carried on anyway, which prosody alone cannot separate — a native Hindi speaker
scored 75% blind on this exact contrast, so roughly 25 points of the gap is
audible headroom and the rest may not be.

**With one more day.** Round two of the error-listening kit (already built, 32
clips of the model's most expensive out-of-fold mistakes) to mine the false
alarms the way round one produced the pitch-percentile and MFCC features; then a
small learned front-end — a shallow CNN over log-mel of the last 1.5 s, trained
from scratch with time-stretch and noise augmentation to fight the 200-turn
sample size — since the handcrafted plateau (runs 8–10) is a feature-vocabulary
ceiling, not a model-capacity one.
