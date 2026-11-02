"""Pull a speaker-balanced sample of clips from In-the-Wild by remote read.

Only the clips we need are fetched, not the full 8 GB archive.
Every log line has a timestamp and done/total progress.
"""
import csv
import io
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from remotezip import RemoteZip

URL = "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
OUT = "work/clips"
PER_LABEL = 30          # clips per speaker per label
MAX_SEC = 6.0           # keep first 6 seconds
MIN_SEC = 2.0
SR = 16000
SEED = 7

# Speakers held out for the final test (never used to train or choose a model)
TEST_SPEAKERS = [
    "Barack Obama", "Donald Trump", "Alec Guinness", "Arnold Schwarzenegger",
    "George Carlin", "Gilbert Gottfried", "Adam Driver", "John Cleese",
    "Bill Burr", "Kanye West", "Mark Zuckerberg", "Winston Churchill",
]
# Speakers used only to choose between models
CHECK_SPEAKERS = [
    "Bernie Sanders", "Ayn Rand", "Bill Clinton", "Ronald Reagan",
    "Christopher Hitchens", "Martin Luther King", "Bob Ross", "2Pac",
]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    random.seed(SEED)
    import os
    os.makedirs(OUT, exist_ok=True)
    with RemoteZip(URL) as z:
        rows = list(csv.DictReader(io.StringIO(z.read("release_in_the_wild/meta.csv").decode())))
        sizes = {i.filename: i.file_size for i in z.infolist()}
    by = {}
    for r in rows:
        f = "release_in_the_wild/" + r["file"]
        if sizes.get(f, 0) < 64000:   # skip very short clips (~<2 s)
            continue
        by.setdefault((r["speaker"], r["label"]), []).append(r["file"])
    speakers = sorted({s for s, _ in by})
    picks = []
    for s in speakers:
        for lab in ("bona-fide", "spoof"):
            files = by.get((s, lab), [])
            random.shuffle(files)
            for f in files[:PER_LABEL]:
                split = "test" if s in TEST_SPEAKERS else "check" if s in CHECK_SPEAKERS else "train"
                picks.append({"file": f, "speaker": s, "label": lab, "split": split})
    total = len(picks)
    log(f"selected {total} clips from {len(speakers)} speakers")
    done = [0]
    lock = threading.Lock()
    local = threading.local()

    def get_zip():
        if not hasattr(local, "z"):
            local.z = RemoteZip(URL)
        return local.z

    def fetch(p):
        out = f"{OUT}/{p['file'].replace('.wav', '.npy')}"
        try:
            b = get_zip().read("release_in_the_wild/" + p["file"])
            x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if sr != SR:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=SR)
            x = x[: int(MAX_SEC * SR)]
            if len(x) < MIN_SEC * SR:
                return None
            np.save(out, (x * 32767).astype(np.int16))
            p = dict(p, npy=out, seconds=round(len(x) / SR, 2))
        except Exception as e:  # keep going, record the failure
            log(f"FAILED {p['file']}: {e}")
            p = None
        with lock:
            done[0] += 1
            if done[0] % 100 == 0 or done[0] == total:
                log(f"fetched {done[0]}/{total}")
        return p

    with ThreadPoolExecutor(8) as ex:
        res = [r for r in ex.map(fetch, picks) if r]
    json.dump(res, open("work/manifest.json", "w"))
    log(f"saved {len(res)} clips; manifest written")
    from collections import Counter
    log(str(Counter((r["split"], r["label"]) for r in res)))


if __name__ == "__main__":
    sys.exit(main())
