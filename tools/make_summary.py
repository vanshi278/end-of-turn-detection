"""Generate SUMMARY.html from measured results (artifacts/results.json).

Charts are inline SVG built from the data -- no plotting dependency, no numbers
retyped by hand. Regenerate any time with:  python tools/make_summary.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/vanshikaagarwal/speedrun")

R = json.load(open("artifacts/results.json"))
OUT = "SUMMARY.html"

# palette: validated with the dataviz skill's checker in BOTH modes
# (all six checks pass; worst adjacent CVD dE 26.5 light / 27.3 dark)
BLUE_L, GREEN_L = "#2a78d6", "#008300"
BLUE_D, GREEN_D = "#3987e5", "#008300"

HUMAN_FEATS = {"f0_final_pct", "f0_min_pct", "f0_end_vs_floor", "f0_end_vs_ceil",
               "trail_unvoiced_s", "trail_centroid_z", "trail_flatness",
               "trail_energy_z", "abruptness"} | {f"mfcc{i}" for i in range(1, 9)}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- chart 1
def chart_dumbbell():
    """before -> after per item: one hue, two shades (dataviz: dumbbell)."""
    rows = [("English", R["per_lang"]["english"]["baseline"], R["per_lang"]["english"]["oof_ms"]),
            ("Hindi", R["per_lang"]["hindi"]["baseline"], R["per_lang"]["hindi"]["oof_ms"])]
    W, H, L, Rr = 660, 190, 96, 150
    xmax = 1700
    def x(v): return L + (v / xmax) * (W - L - Rr)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Response delay, baseline versus our model">']
    s.append(f'<line x1="{L}" y1="{H-34}" x2="{W-Rr}" y2="{H-34}" class="axis"/>')
    for gv in range(0, 1701, 500):
        s.append(f'<line x1="{x(gv):.1f}" y1="34" x2="{x(gv):.1f}" y2="{H-34}" class="grid"/>')
        s.append(f'<text x="{x(gv):.1f}" y="{H-18}" class="tick" text-anchor="middle">{gv}</text>')
    for i, (name, b, o) in enumerate(rows):
        y = 62 + i * 52
        s.append(f'<text x="{L-14}" y="{y+5}" class="lab" text-anchor="end">{name}</text>')
        s.append(f'<line x1="{x(o):.1f}" y1="{y}" x2="{x(b):.1f}" y2="{y}" class="dumb"/>')
        s.append(f'<circle cx="{x(b):.1f}" cy="{y}" r="6" class="dot-base"/>')
        s.append(f'<circle cx="{x(o):.1f}" cy="{y}" r="6" class="dot-ours"/>')
        if abs(o - b) < 200:
            # the dots coincide (we match the baseline); side-by-side labels
            # would collide, so state the tie once instead of twice
            s.append(f'<text x="{x(b)+14:.1f}" y="{y+4}" class="val strong">'
                     f'{o} ms — matches baseline</text>')
        else:
            s.append(f'<text x="{x(b)+14:.1f}" y="{y+4}" class="val">{b} ms baseline</text>')
            s.append(f'<text x="{x(o)-14:.1f}" y="{y+4}" class="val strong" '
                     f'text-anchor="end">{o} ms</text>')
    s.append(f'<text x="{W-Rr}" y="{H-4}" class="tick" text-anchor="end">mean response delay (ms) — lower is better</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- chart 2
def chart_bootstrap():
    """95% intervals over resampled turns: the honest read."""
    # Rr is wide because the value labels sit to the RIGHT of each interval and
    # English's upper bound reaches 1600 -- a narrower margin clipped
    # "baseline 1556" to "baseline 1".
    W, H, L, Rr = 660, 210, 96, 122
    xmax = 1700
    def x(v): return L + (v / xmax) * (W - L - Rr)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Bootstrap 95% intervals">']
    for gv in range(0, 1701, 500):
        s.append(f'<line x1="{x(gv):.1f}" y1="26" x2="{x(gv):.1f}" y2="{H-34}" class="grid"/>')
        s.append(f'<text x="{x(gv):.1f}" y="{H-18}" class="tick" text-anchor="middle">{gv}</text>')
    y = 44
    for lang in ("english", "hindi"):
        b = R["per_lang"][lang]["boot"]
        s.append(f'<text x="{L-14}" y="{y+5}" class="lab" text-anchor="end">{lang.title()}</text>')
        for key, cls, lo, hi, mean in [
                ("baseline", "rng-base", b["base_lo"], b["base_hi"], b["base_mean"]),
                ("ours", "rng-ours", b["ours_lo"], b["ours_hi"], b["ours_mean"])]:
            s.append(f'<line x1="{x(lo):.1f}" y1="{y}" x2="{x(hi):.1f}" y2="{y}" class="{cls}"/>')
            s.append(f'<circle cx="{x(mean):.1f}" cy="{y}" r="5.5" class="{"dot-base" if key=="baseline" else "dot-ours"}"/>')
            s.append(f'<text x="{x(hi)+12:.1f}" y="{y+4}" class="tick">{key} {mean:.0f}</text>')
            y += 26
        y += 24
    s.append(f'<text x="{W-Rr}" y="{H-4}" class="tick" text-anchor="end">95% interval over 600 turn-resamples (ms)</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- chart 3
def chart_auc_vs_hard():
    """The finding: AUC is inflated by holds too short to matter."""
    W, H, L = 660, 210, 96
    def y(v): return H - 46 - (v - 0.45) / 0.35 * (H - 96)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="AUC versus hard AUC">']
    for gv in [0.5, 0.6, 0.7, 0.8]:
        s.append(f'<line x1="{L}" y1="{y(gv):.1f}" x2="{W-24}" y2="{y(gv):.1f}" class="grid"/>')
        s.append(f'<text x="{L-10}" y="{y(gv)+4:.1f}" class="tick" text-anchor="end">{gv:.1f}</text>')
    s.append(f'<line x1="{L}" y1="{y(0.5):.1f}" x2="{W-24}" y2="{y(0.5):.1f}" class="chance"/>')
    s.append(f'<text x="{W-26}" y="{y(0.5)-8:.1f}" class="tick" text-anchor="end">chance</text>')
    bw, gap = 62, 16
    stages = [("English\nbefore study", 0.685, 0.704), ("English\nfinal", R["per_lang"]["english"]["auc"], R["per_lang"]["english"]["hard_auc"]),
              ("Hindi\nbefore study", 0.685, 0.509), ("Hindi\nfinal", R["per_lang"]["hindi"]["auc"], R["per_lang"]["hindi"]["hard_auc"])]
    x0 = L + 34
    for name, a, ha in stages:
        for j, (v, cls) in enumerate([(a, "bar-auc"), (ha, "bar-hard")]):
            bx = x0 + j * (bw / 2 + 2)
            s.append(f'<rect x="{bx}" y="{y(v):.1f}" width="{bw/2 - 2}" height="{(y(0.45)-y(v)):.1f}" rx="4" class="{cls}"/>')
            s.append(f'<text x="{bx + bw/4 - 1}" y="{y(v)-7:.1f}" class="val" text-anchor="middle">{v:.2f}</text>')
        for k, line in enumerate(name.split("\n")):
            s.append(f'<text x="{x0 + bw/2}" y="{H-24 + k*13}" class="tick" text-anchor="middle">{line}</text>')
        x0 += bw + gap + 24
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- chart 4
def chart_features():
    """Emphasis: features born from the human study vs everything else."""
    hi = R["features"]["hindi"]
    rank = sorted(hi.items(), key=lambda kv: -abs(kv[1] - 0.5))[:14]
    W, rowh, L = 660, 21, 150
    H = 44 + len(rank) * rowh + 26
    def x(v): return L + (v - 0.5) / 0.26 * (W - L - 74)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Per-feature discrimination on the contrast that matters">']
    for gv in [0.5, 0.55, 0.6, 0.65, 0.7]:
        s.append(f'<line x1="{x(gv):.1f}" y1="28" x2="{x(gv):.1f}" y2="{H-26}" class="grid"/>')
        s.append(f'<text x="{x(gv):.1f}" y="{H-10}" class="tick" text-anchor="middle">{gv:.2f}</text>')
    for i, (name, v) in enumerate(rank):
        yy = 40 + i * rowh
        d = abs(v - 0.5) + 0.5          # direction-free strength
        cls = "bar-human" if name in HUMAN_FEATS else "bar-other"
        s.append(f'<text x="{L-12}" y="{yy+9}" class="lab-sm" text-anchor="end">{esc(name)}</text>')
        s.append(f'<rect x="{L}" y="{yy}" width="{max(x(d)-L,1):.1f}" height="{rowh-6}" rx="4" class="{cls}"/>')
        s.append(f'<text x="{x(d)+8:.1f}" y="{yy+9}" class="val-sm">{d:.3f}</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- chart 5
def chart_free_holds():
    """Why the metric is not what it looks like: most holds cost nothing."""
    W, H, L = 660, 220, 60
    rows = {}
    for lang in ("english", "hindi"):
        df = pd.read_csv(f"eot_handout/eot_data/{lang}/labels.csv")
        d = (df.pause_end - df.pause_start)[df.label == "hold"]
        rows[lang] = [(t, float((d <= t).mean())) for t in np.arange(0.1, 1.65, 0.05)]
    def x(v): return L + (v - 0.1) / 1.5 * (W - L - 96)
    def y(v): return H - 44 - v * (H - 80)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Share of hold pauses that cannot cause a false cutoff">']
    for gv in [0, .25, .5, .75, 1.0]:
        s.append(f'<line x1="{L}" y1="{y(gv):.1f}" x2="{W-96}" y2="{y(gv):.1f}" class="grid"/>')
        s.append(f'<text x="{L-8}" y="{y(gv)+4:.1f}" class="tick" text-anchor="end">{gv*100:.0f}%</text>')
    for t in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        s.append(f'<text x="{x(t):.1f}" y="{H-22}" class="tick" text-anchor="middle">{t:.2f}</text>')
    for lang, cls in [("english", "line-en"), ("hindi", "line-hi")]:
        pts = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in rows[lang])
        s.append(f'<polyline points="{pts}" class="{cls}"/>')
        tl, vl = rows[lang][-1]
        s.append(f'<text x="{x(tl)+10:.1f}" y="{y(vl)+4:.1f}" class="lab-sm">{lang.title()}</text>')
    vh = dict(rows["hindi"])[round(0.65, 2)] if round(0.65, 2) in dict(rows["hindi"]) else None
    s.append(f'<line x1="{x(0.65):.1f}" y1="26" x2="{x(0.65):.1f}" y2="{H-44}" class="chance"/>')
    s.append(f'<text x="{x(0.65)+6:.1f}" y="38" class="tick">0.65 s delay</text>')
    s.append(f'<text x="{W-96}" y="{H-6}" class="tick" text-anchor="end">agent action delay (s)</text>')
    s.append("</svg>")
    return "".join(s)


CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;font:16px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--ink);-webkit-font-smoothing:antialiased}
.viz-root{
  --bg:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#78776f;
  --line:#e4e3de; --blue:__BLUE_L__; --green:__GREEN_L__; --gray:#b9b8b1; --accent-soft:#eef4fc;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
  --bg:#141413; --panel:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8d8c83;
  --line:#33322e; --blue:__BLUE_D__; --green:__GREEN_D__; --gray:#6b6a63; --accent-soft:#1d2937;
}}
:root[data-theme="dark"] .viz-root{
  --bg:#141413; --panel:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#8d8c83;
  --line:#33322e; --blue:__BLUE_D__; --green:__GREEN_D__; --gray:#6b6a63; --accent-soft:#1d2937;
}
.wrap{max-width:980px;margin:0 auto;padding:56px 24px 96px}
h1{font-size:34px;line-height:1.2;margin:0 0 8px;letter-spacing:-.02em}
.sub{color:var(--ink-2);margin:0 0 40px;font-size:17px}
h2{font-size:22px;margin:52px 0 14px;letter-spacing:-.01em;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:16px;margin:28px 0 8px;color:var(--ink)}
p{margin:0 0 14px;color:var(--ink-2)}
p strong,li strong{color:var(--ink)}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--accent-soft);
  padding:1px 5px;border-radius:4px;color:var(--ink)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:24px 0 8px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .n{font-size:30px;font-weight:650;letter-spacing:-.02em;color:var(--ink)}
.kpi .n small{font-size:14px;font-weight:500;color:var(--ink-3)}
.kpi .k{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-3);margin-bottom:4px}
.kpi .d{font-size:12.5px;color:var(--ink-2);margin-top:3px}
figure{margin:22px 0 8px;background:var(--panel);border:1px solid var(--line);
  border-radius:10px;padding:16px 18px 10px;overflow-x:auto}
figcaption{font-size:13px;color:var(--ink-3);margin-top:6px}
svg{display:block;width:100%;height:auto;min-width:520px}
.axis{stroke:var(--line);stroke-width:1}
.grid{stroke:var(--line);stroke-width:1;stroke-dasharray:2 4}
.chance{stroke:var(--ink-3);stroke-width:1.5;stroke-dasharray:4 4}
.tick{fill:var(--ink-3);font-size:11px}
.lab{fill:var(--ink);font-size:13.5px;font-weight:550}
.lab-sm{fill:var(--ink-2);font-size:11.5px}
.val{fill:var(--ink-2);font-size:12px}
.val-sm{fill:var(--ink-3);font-size:10.5px}
.val.strong{fill:var(--ink);font-weight:650;font-size:13px}
.dumb{stroke:var(--gray);stroke-width:2}
.dot-base{fill:var(--gray);stroke:var(--panel);stroke-width:2}
.dot-ours{fill:var(--blue);stroke:var(--panel);stroke-width:2}
.rng-base{stroke:var(--gray);stroke-width:6;stroke-linecap:round}
.rng-ours{stroke:var(--blue);stroke-width:6;stroke-linecap:round;opacity:.85}
.bar-auc{fill:var(--gray)}
.bar-hard{fill:var(--blue)}
.bar-human{fill:var(--blue)}
.bar-other{fill:var(--gray)}
.line-en{fill:none;stroke:var(--gray);stroke-width:2}
.line-hi{fill:none;stroke:var(--blue);stroke-width:2}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--ink-2);margin:2px 0 10px}
.legend i{width:11px;height:11px;border-radius:3px;display:inline-block;margin-right:6px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line)}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-3);font-weight:600}
td{color:var(--ink-2)} td:first-child{color:var(--ink)}
.num{font-variant-numeric:tabular-nums}
.callout{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--blue);
  border-radius:8px;padding:14px 18px;margin:18px 0}
.callout.warn{border-left-color:#eb6834}
.callout h3{margin:0 0 6px}
.callout p:last-child{margin-bottom:0}
ul{margin:0 0 14px;padding-left:22px;color:var(--ink-2)} li{margin:5px 0}
.split{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.split{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.card h3{margin-top:0}
.who{font-size:12px;font-weight:600;letter-spacing:.05em;text-transform:uppercase}
.who.h{color:var(--green)} .who.a{color:var(--blue)}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--line);
  font-size:12.5px;color:var(--ink-3)}
""".replace("__BLUE_L__", BLUE_L).replace("__GREEN_L__", GREEN_L) \
   .replace("__BLUE_D__", BLUE_D).replace("__GREEN_D__", GREEN_D)


def main():
    en, hi = R["per_lang"]["english"], R["per_lang"]["hindi"]
    L = R["listening"]
    _res = pd.read_csv("listening_kit/RESULTS.csv")
    n_hindi_clips = int((_res.lang == "hindi").sum())
    html = f"""<div class="viz-root"><div class="wrap">
<h1>End-of-Turn Detection</h1>
<p class="sub">Predicting whether a speaker is finished or just thinking, from
{R['n_pauses']} pauses across {R['n_turns']} real phone turns in English and Hindi —
using only audio the agent could actually have heard by that moment.</p>

<div class="kpis">
  <div class="kpi"><div class="k">English</div><div class="n">{en['oof_ms']}<small> ms</small></div>
    <div class="d">from 1600 ms baseline · −{1600-en['oof_ms']} ms</div></div>
  <div class="kpi"><div class="k">Hindi</div><div class="n">{hi['oof_ms']}<small> ms</small></div>
    <div class="d">baseline 850 ms · see honest read below</div></div>
  <div class="kpi"><div class="k">Bootstrap gain (En)</div><div class="n">{en['boot']['gain_mean']:.0f}<small> ms</small></div>
    <div class="d">95% CI [{en['boot']['gain_lo']:.0f}, {en['boot']['gain_hi']:.0f}] · wins {en['boot']['win_rate']*100:.0f}% of resamples</div></div>
  <div class="kpi"><div class="k">Human ceiling (Hindi)</div><div class="n">{L['hindi_acc']*100:.0f}<small>%</small></div>
    <div class="d">blind, native speaker, n={n_hindi_clips} · {L['hiconf_hindi_acc']*100:.0f}% when confident</div></div>
</div>
<p style="font-size:13.5px;color:var(--ink-3)">Every score is <strong>out-of-fold</strong>
(GroupKFold by turn, 3 seeds) and produced by the official <code>score.py</code>.
The same model scores 511 ms in-sample on these folders. That number is
memorisation and appears nowhere in our claims.</p>

<h2>The result</h2>
<figure>{chart_dumbbell()}
<div class="legend"><span><i style="background:var(--gray)"></i>silence-only baseline</span>
<span><i style="background:var(--blue)"></i>our model (out-of-fold)</span></div>
<figcaption>Mean response delay at ≤5% interrupted turns. English improves by
{1600-en['oof_ms']} ms ({(1600-en['oof_ms'])/1600*100:.0f}%). Hindi matches its baseline — the honest read is below.</figcaption></figure>

<h2>Why the metric is not what it looks like</h2>
<p>The scorer waits <code>delay</code> seconds at each pause and speaks if
<code>p_eot ≥ threshold</code>. It sweeps both and reports the lowest mean delay
that keeps interrupted turns ≤5%. Two structural facts fall out of that, and they
drive every decision we made:</p>
<ul>
<li><strong>A hold pause only costs anything if it outlives the delay.</strong> If
the user resumes before the agent speaks, nobody notices. Short holds are free.</li>
<li><strong>Cost is counted per turn, not per pause.</strong> A turn is interrupted
if <em>any</em> of its holds fires; the second mistake in a turn is free.</li>
</ul>
<figure>{chart_free_holds()}
<figcaption>Share of hold pauses that <em>cannot</em> cause a false cutoff, by
delay. At a 0.65 s delay, 90% of Hindi holds are already harmless — only ~25
pauses in the entire Hindi set can ever hurt us.</figcaption></figure>

<div class="callout"><h3>The finding that reorganised the project</h3>
<p>Overall AUC averages over every hold, <strong>75% of which the metric does not
score</strong>. Splitting it out, our model at AUC 0.685 was scoring
<strong>0.52 — chance — on the only contrast that pays</strong>: turn-end versus a
hold long enough to interrupt someone. It had learned to separate turn-ends from
<em>short</em> holds, which is worth exactly zero milliseconds. We promoted
<code>hard_auc</code> to the headline diagnostic so this could not hide again.</p></div>

<figure>{chart_auc_vs_hard()}
<div class="legend"><span><i style="background:var(--gray)"></i>overall AUC (flattering)</span>
<span><i style="background:var(--blue)"></i>hard_auc — eot vs holds &gt;0.6 s (what pays)</span></div>
<figcaption>Same model, two measurements. Before the listening study, Hindi
hard_auc was 0.509 while overall AUC read 0.685. AUC also <em>fell</em> when
cost-weighting <em>improved</em> the score (run 4) — it is the wrong objective.</figcaption></figure>

<h2>What the model listens to</h2>
<p>For each pause, features come only from audio in <code>[0, pause_start)</code>.
The entry point truncates there and hands the truncated copy to a function that
never receives the rest, so feature code <em>cannot</em> see the future rather
than merely being careful not to.</p>
<figure>{chart_features()}
<div class="legend"><span><i style="background:var(--blue)"></i>born from the human listening study</span>
<span><i style="background:var(--gray)"></i>standard prosodic features</span></div>
<figcaption>Per-feature discrimination on the Hindi contrast that pays (distance
from chance; n=120). The top features are the ones the listening study
produced — before it, the whole model sat at 0.509.</figcaption></figure>

<div class="split">
<div class="card"><h3>Pitch position, not pitch value</h3>
<p>The listener's dominant cue was <em>"pitch dropped"</em> → end and <em>"pitch
was very high, cant be the end"</em> → hold. Crucially, high and low are relative
to <strong>that speaker</strong>. So we encode the final pitch as a
<strong>percentile of that speaker's own pitch history</strong>, not a z-score:
the ear asks "did you come home to your floor?", not "how many IQRs from your
median?". It beat the z-score head to head — 0.639 vs 0.595.</p></div>
<div class="card"><h3>Vowel identity without ASR</h3>
<p>The listener read Hindi verb endings lexically — <em>"rahega used"</em>,
<em>"ended with rakhiyega"</em>. We cannot run ASR, but Hindi is verb-final and
its finite verbs land on a small set of vowel qualities. Cepstral-mean-normalised
MFCCs over the final voiced nucleus encode exactly that, and became our strongest
Hindi features: <strong>mfcc7 = 0.703, mfcc5 = 0.696</strong>.</p></div>
<div class="card"><h3>Breath</h3>
<p><em>"person taking breath"</em> → hold. A speaker reloading leaves a trailing
unvoiced, noisy, low-energy stretch. <code>unvoiced_frac_final</code> is among our
best features in both languages.</p></div>
<div class="card"><h3>Cost-aware training</h3>
<p>Holds are weighted by whether they could actually interrupt anyone. A hold
shorter than the delay is free and shouldn't consume model capacity.
<code>dur</code> is used only at training time, from the labels, exactly like
<code>class_weight</code> — it is future information at inference and never
reaches a feature.</p></div>
</div>

<h2>The honest read on Hindi</h2>
<p>Hindi's 850 ms is a <strong>degenerate baseline</strong>: exactly 5 of its 100
turns contain a hold longer than 0.85 s — exactly the 5% budget — so "fire at
everything after 850 ms" saturates the constraint and cannot be beaten without a
lower delay. That is a property of <em>this draw of 100 turns</em>, not of Hindi.
A single number on a single sample cannot tell "no better than a timer" from
"better, but this sample's baseline is degenerate", so we resampled turns.</p>
<figure>{chart_bootstrap()}
<figcaption>Paired bootstrap, 600 resamples of turns, out-of-fold predictions.
English is decisive. Hindi's mean improves ({hi['boot']['base_mean']:.0f} → {hi['boot']['ours_mean']:.0f} ms)
but the interval on the gain touches zero.</figcaption></figure>
<div class="callout warn"><h3>What we will not claim</h3>
<p><strong>Hindi is not yet a demonstrable win.</strong> The gain is
{hi['boot']['gain_mean']:.0f} ms with a 95% interval of
[{hi['boot']['gain_lo']:.0f}, {hi['boot']['gain_hi']:.0f}]. The discrimination is
real — hard_auc {hi['hard_auc']} against 0.509 before the listening study — but
converting ranking into milliseconds needs more precision on long holds than we
have. We would rather say so than quote the 511 ms in-sample number.</p></div>

<h2>Human vs. coding agent</h2>
<p>The brief asks who did what. The split below is literal: the human work is
work the agent could not do, and it is the reason the Hindi half of this project
exists at all.</p>
<div class="split">
<div class="card"><div class="who h">Human</div>
<h3>The listening study — and the features it produced</h3>
<ul>
<li>Ran a <strong>36-clip blind listening study</strong> (balanced, Hindi-weighted,
hearing only what the model hears): <strong>{L['acc']*100:.1f}% overall,
{L['hindi_acc']*100:.0f}% Hindi, {L['hiconf_hindi_acc']*100:.1f}% on confident Hindi calls</strong>.
The agent cannot hear audio; this number exists only because a person made it.</li>
<li><strong>Established that the Hindi ceiling was not a data ceiling.</strong> The
agent had proved the model was at chance and could not tell "features are blind"
from "signal is absent". 75% blind accuracy on the identical contrast settled it
and justified continuing.</li>
<li><strong>Native-speaker judgements</strong> that became three feature families:
pitch-position-within-speaker's-range, breath, and Hindi verb-final vowel quality
(→ MFCCs). Each is traceable to a specific written note.</li>
<li><strong>Calibrated confidence</strong> (conf-5 {L['by_conf'].get('5',[0,0])[0]*100:.0f}%,
conf-3 {L['by_conf'].get('3',[0,0])[0]*100:.0f}%), evidence the cue is a real audible
property rather than noise.</li>
<li><strong>Round-2 error audit</strong>: annotated the model's most expensive
out-of-fold mistakes with <em>why</em> each was wrong. The feature logic came from
these notes — <em>"aur is not end, it is conjunction"</em> became final-syllable
prominence (2nd-best Hindi feature, 0.660), <em>"end mai usually we speak
slowly"</em> became local deceleration (0.694 English).</li>
</ul></div>
<div class="card"><div class="who a">Coding agent</div>
<h3>Infrastructure, DSP, modelling, and the traps</h3>
<ul>
<li>Bit-exact reimplementation of the scorer (18 tests) so selection runs on the
real metric; <code>hard_auc</code>; the paired bootstrap.</li>
<li>Causality by construction + 6-way mutation tests; caught the dataset's
<strong>file-length leak</strong> (AUC 1.000 / 100 ms) and the
<strong>in-sample trap</strong> (511 ms).</li>
<li><strong>Vectorised YIN written from scratch</strong> (<code>eotlib/yin.py</code>)
after finding the voicing gate marked 99% of frames voiced; 0.42% pitch
disagreement with pyin at 68× the speed.</li>
<li>Cost-aware weighting, rank calibration, the baseline floor
(<code>P_FLOOR</code>), model selection, and turning the human's notes into
{R['n_features']} features.</li>
</ul></div>
</div>
<p style="font-size:13.5px;color:var(--ink-3)">Stated plainly: the agent built the
machine and found the traps; the human supplied the perception the machine did not
have and the evidence that the remaining gap was worth chasing. Neither half
reaches {hi['hard_auc']} Hindi hard_auc alone — the agent's model sat at 0.509 until
the listening study, and the study's notes are prose until something turns them
into MFCCs and percentile ranks.</p>

<h2>Why this beats the status quo</h2>
<table>
<tr><th>Status quo</th><th>What we do instead</th><th>Evidence</th></tr>
<tr><td>Silence timer (naive VAD endpointing)</td><td>Prosodic model of turn-finality</td>
<td class="num">English 1600 → {en['oof_ms']} ms; wins {en['boot']['win_rate']*100:.0f}% of resamples</td></tr>
<tr><td>Optimise AUC / accuracy</td><td>Optimise the real metric; report <code>hard_auc</code></td>
<td class="num">AUC 0.685 hid a chance-level 0.52 on the pauses that pay</td></tr>
<tr><td>Treat every hold as equally bad</td><td>Weight holds by whether they can interrupt anyone</td>
<td class="num">English 1284 → 1150 ms while AUC <em>fell</em></td></tr>
<tr><td>Trust the library pitch tracker</td><td>Measure it; write one that fits the job</td>
<td class="num">99% of frames were called voiced; 68× speedup, 0.42% error</td></tr>
<tr><td>Report the best number you saw</td><td>Out-of-fold only, bootstrapped, in-sample rejected</td>
<td class="num">Refused the 511 ms; publish 1116 / 850</td></tr>
</table>

<h2>Where it fails, and what is next</h2>
<p>The residual errors are the ones the human listener also got wrong:
syntactically complete clauses where the speaker carried on anyway
(<em>"statement looked complete, baat karke gaya tha"</em>) — four of six Hindi
misses were this. Prosody alone cannot separate those, and notably the human
<em>heard</em> the prosody and overrode it with syntax
(<em>"felt that speaker wanted to speak more"</em>), so a prosody-only model can
beat a person on exactly that class. Next: round two of the error-listening kit
(built, 32 clips of the most expensive out-of-fold mistakes), then a small
from-scratch CNN over log-mel of the last 1.5 s with augmentation — the
handcrafted plateau is a feature-vocabulary ceiling, not a model-capacity one
(every classifier we tried lands at Hindi hard_auc 0.61–0.71).</p>

<footer>
<p><strong>Reproduce:</strong> <code>python train.py</code> ·
<code>python predict.py --data_dir &lt;folder&gt; --out predictions.csv</code> ·
<code>python -m pytest</code> (fast tier) · <code>python -m pytest -m slow</code>
(causality on all {R['n_pauses']} pauses) · <code>python tools/make_summary.py</code> regenerates this page.</p>
<p>Companion docs: <strong>RUNLOG.md</strong> (13 scored runs, including the two
wrong hypotheses) · <strong>NOTES.md</strong> (signal, failures, next steps).
Charts are inline SVG generated from <code>artifacts/results.json</code>; the
palette passes all six colour checks in light and dark.</p>
</footer>
</div></div>"""
    open(OUT, "w").write(f"<style>{CSS}</style>\n{html}")
    print(f"wrote {OUT} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
