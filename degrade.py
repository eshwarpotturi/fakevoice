"""Make phone-quality copies of the check and test clips.

Phone line model: band-limit to 300-3400 Hz, 8 kHz sampling, G.711 mu-law
compression, then back to 16 kHz. Written to work/clips_phone/ and
work/manifest_phone.json.
"""
import json
import os
import subprocess
import time

import numpy as np

SR = 16000


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def phone(x16):
    """x16: float32 mono at 16 kHz -> float32 mono at 16 kHz, phone quality."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
        "-af", "highpass=f=300,lowpass=f=3400,aresample=8000",
        "-ar", "8000", "-c:a", "pcm_mulaw", "-f", "mulaw", "pipe:1",
    ]
    enc = subprocess.run(cmd, input=x16.tobytes(), capture_output=True, check=True).stdout
    cmd2 = ["ffmpeg", "-loglevel", "error", "-f", "mulaw", "-ar", "8000", "-ac", "1", "-i", "pipe:0",
            "-ar", "16000", "-f", "f32le", "pipe:1"]
    dec = subprocess.run(cmd2, input=enc, capture_output=True, check=True).stdout
    return np.frombuffer(dec, dtype=np.float32)


def main():
    import sys
    splits = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("check", "test")
    outname = sys.argv[2] if len(sys.argv) > 2 else "work/manifest_phone.json"
    manifest = json.load(open("work/manifest.json"))
    os.makedirs("work/clips_phone", exist_ok=True)
    out = []
    sel = [m for m in manifest if m["split"] in splits]
    for i, m in enumerate(sel, 1):
        x = np.load(m["npy"]).astype(np.float32) / 32767.0
        y = phone(x)
        path = m["npy"].replace("work/clips/", "work/clips_phone/")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, (np.clip(y, -1, 1) * 32767).astype(np.int16))
        out.append(dict(m, npy=path))
        if i % 100 == 0 or i == len(sel):
            log(f"degraded {i}/{len(sel)}")
    json.dump(out, open(outname, "w"))
    log("done")


if __name__ == "__main__":
    main()
