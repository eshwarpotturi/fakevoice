"""Turn work/results2.json into the files the website reads."""
import json

import numpy as np

R = json.load(open("work/results2.json"))
H = np.load("work/heads.npz")
simple = json.load(open("work/shortcut.json")) if __import__("os").path.exists("work/shortcut.json") else None

USED = "wavlm_both"
names = {
    "wavlm_clean": "WavLM, trained on clean audio",
    "wavlm_both": "WavLM, trained on clean and phone-quality audio (used in the demo)",
    "w2v2_clean": "wav2vec 2.0, trained on clean audio",
    "gemini_clean": "Gemini Embedding 2, trained on clean audio",
}
for d in R["detectors"]:
    if d["id"] in names:
        d["name"] = names[d["id"]]
    d.pop("platt", None)
    d["check_clean_auc"] = round(d.get("check_clean_auc", 0), 4)

used = next(d for d in R["detectors"] if d["id"] == USED)
_cal = json.load(open("work/calibration.json"))
used["clean"]["accuracy"], used["phone"]["accuracy"] = _cal["acc_clean"], _cal["acc_phone"]   # at the demo's 50% point
ready = [d for d in R["detectors"] if d["kind"] == "off-the-shelf"]
c = R["counts"]
pc = lambda v: f"{round(v * 100)}%"

R["best"] = USED
R["headline"] = (f"Our detector was right on {pc(used['clean']['accuracy'])} of clips on clean audio and "
                 f"{pc(used['phone']['accuracy'])} over a simulated phone line, on people it had never heard")
if ready:
    r0 = ready[0]
    R["subhead"] = (f"A ready-made detector from the Hugging Face model hub was right on only {pc(r0['clean']['accuracy'])} of the same clips. "
                    f"Tested on {c['test']} clips from {c['test_speakers']} people, half real and half AI-generated.")
else:
    R["subhead"] = f"Tested on {c['test']} clips from {c['test_speakers']} people, half real and half AI-generated."

tr_sp = c["train_speakers"]
R["method_html"] = f"""
<ol>
<li><b>Data.</b> Real and AI-faked recordings of 54 public figures, gathered from the internet by researchers. We made no fake audio.</li>
<li><b>Sample.</b> Up to 30 real and 30 fake 6-second clips per person: {c['train'] + c['check'] + c['test']:,} clips.</li>
<li><b>Split by person.</b> {tr_sp} people to train, {c['check_speakers']} to choose settings, {c['test_speakers']} held back for the test.</li>
<li><b>Detectors.</b> WavLM, wav2vec 2.0 and Gemini Embedding 2 listen; a small classifier learns real against fake. One ready-made detector was tested as downloaded.</li>
<li><b>Phone line.</b> 300 to 3,400 Hz, 8 kHz, standard phone compression.</li>
</ol>"""

simple_txt = f" A loudness-only check scores {simple['test']:.2f} (0.5 is guessing)." if simple else ""
R["caveats_html"] = f"""
<ul>
<li><b>Recording differences.</b> Real and fake clips come from different sources, so part of the score may reflect the recording, not the voice.{simple_txt}</li>
<li><b>One collection.</b> English-speaking public figures. New voice generators, Indian voices, other languages and real calls are untested.</li>
<li><b>Modern clones.</b> A small test of three open-model clones of one person's voice: all 3 caught on clean audio, 1 to 2 of 3 over a simulated phone line.</li>
<li><b>Real calls are mostly genuine.</b> If 1 call in 1,000 is fake, a 5% false-alarm rate flags about 50 genuine callers per fake caught.</li>
<li><b>Few people.</b> {c['test_speakers']} test people, about {round(c['test'] / c['test_speakers'])} clips each.</li>
<li><b>A signal, not proof.</b> Use it with other checks, such as a call-back.</li>
</ul>"""

CAL = json.load(open("work/calibration.json"))
R["calibration"] = {"test_bins": CAL["test_bins"], "note": "When the detector sounded sure a clip was human (under 5%), about 6 in 100 were actually fake. The demo therefore never shows less than 5% AI. When it sounded sure a clip was fake (over 95%), all of them were."}
R["fingerprint"] = json.load(open("work/fingerprint.json"))
json.dump(R, open("docs/data/results.json", "w"), indent=1)

cfg = {
    "window_sec": 6, "max_windows": 12, "floor": 0.05, "ceil": 0.99,
    "platt": {"coef": float(H[USED + "__platt_coef"]), "intercept": float(H[USED + "__platt_intercept"]), "threshold": float(H[USED + "__thr"])},
    "reliability_html": (f"<b>How reliable is this?</b> On {c['test_speakers']} people it had never heard, it was right on {pc(used['clean']['accuracy'])} of clean clips "
                         f"and {pc(used['phone']['accuracy'])} over a simulated phone line. It can still be fooled, and it has not been tested on Indian voices or languages other than English. "
                         "Treat the percentage as one signal, not proof. <a href=\"stats.html\">See how well it works</a>."),
}
json.dump(cfg, open("docs/model/config.json", "w"), indent=1)
print("wrote docs/data/results.json and docs/model/config.json")
print(json.dumps(cfg["platt"]))
