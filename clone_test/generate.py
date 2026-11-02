"""Make new sentences in a voice, from a short recording of that voice.

Use ONLY a voice you own or have the speaker's permission to copy.
Runs on CPU with the open Chatterbox model (about 40 to 60 seconds per short sentence).
Every log line has a timestamp and progress.

Usage:
  python generate.py --reference my_voice.wav --out clones/
Install in its own environment (the detector does not need it):
  python3 -m venv venv
  ./venv/bin/pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
  ./venv/bin/pip install chatterbox-tts
"""
import argparse
import os
import time

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

LINES = [
    "Hello, this is Eswar. I am calling to confirm my appointment for tomorrow afternoon.",
    "Thank you for the update. I will review the report and get back to you in the morning.",
    "Please note my new email address, and let me know once you have received it.",
]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="30 seconds of the speaker's own voice (WAV)")
    ap.add_argument("--out", default="clones")
    ap.add_argument("--lines", nargs="*", default=LINES)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 2)
    log("loading the model")
    model = ChatterboxTTS.from_pretrained(device="cpu")
    log("model loaded")
    for i, t in enumerate(a.lines, 1):
        t0 = time.time()
        wav = model.generate(t, audio_prompt_path=a.reference, exaggeration=0.5, cfg_weight=0.5)
        ta.save(f"{a.out}/clone_{i}.wav", wav, model.sr)
        log(f"made clip {i}/{len(a.lines)} ({wav.shape[-1] / model.sr:.1f} s of audio in {time.time() - t0:.0f} s)")
    log("done")


if __name__ == "__main__":
    main()
