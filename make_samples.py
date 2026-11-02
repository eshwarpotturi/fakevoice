"""Pick demo clips at random (fixed seed) from the TEST speakers.

Clips are not chosen by detector result. File names are neutral (clip-a.wav ...)
so the answer is not in the address; the answer is in data/samples.json and is
only shown after the check.
"""
import json
import random

import numpy as np
import soundfile as sf

random.seed(11)
man = json.load(open("work/manifest.json"))
people = ["Barack Obama", "Donald Trump", "Arnold Schwarzenegger", "John Cleese", "Alec Guinness"]
picks = []
for p in people:
    for lab in ("bona-fide", "spoof"):
        c = [m for m in man if m["split"] == "test" and m["speaker"] == p and m["label"] == lab and m["seconds"] >= 5.0]
        picks.append((p, lab, random.choice(c)))
random.shuffle(picks)    # mix real and fake so the order gives nothing away
out, public = [], []
for i, (p, lab, m) in enumerate(picks):
    letter = chr(97 + i)
    fn = f"data/samples/clip-{letter}.wav"
    x = np.load(m["npy"]).astype(np.float32) / 32767
    sf.write("site/" + fn, x, 16000, subtype="PCM_16")
    truth = "real" if lab == "bona-fide" else "fake"
    public.append({"file": fn, "name": f"{p} (clip {letter.upper()})", "truth": truth})
    out.append({"file": fn, "source": m["file"], "speaker": p, "truth": truth})
json.dump(public, open("site/data/samples.json", "w"), indent=1)
json.dump(out, open("work/samples_source.json", "w"), indent=1)
print([(o["speaker"], o["truth"]) for o in out])
