"""Embed every clip with Gemini Embedding 2 through the company's LLM Foundry gateway.

Clean clips (all) and phone-quality clips (check and test only). The key is read from a file
and never printed. Every log line has a timestamp and done/total. Cost is tracked from the
token counts the service returns.
"""
import base64, io, json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, soundfile as sf

BASE = "https://llmfoundry.straivedemo.com/gemini/v1beta/models/gemini-embedding-2:embedContent"
TOKEN = open(os.environ["FOUNDRY_TOKEN_FILE"]).read().strip()
CAP_RS = 250                      # stop if the high-side cost estimate passes this (Rs, 1 USD = Rs 100)
HIGH_USD_PER_MTOK = 6.50          # higher of the two published prices
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

jobs = []
for m in json.load(open("work/manifest.json")): jobs.append(("clean", m))
for m in json.load(open("work/manifest_phone.json")): jobs.append(("phone", m))
OUT = "work/gemini"; os.makedirs(OUT, exist_ok=True)
state = {"done": 0, "tokens": 0, "fail": 0, "stop": False}; lock = threading.Lock()

def one(job):
    kind, m = job
    path = f"{OUT}/{kind}_{m['file'].replace('.wav', '')}.npy"
    if os.path.exists(path): 
        with lock: state["done"] += 1
        return
    if state["stop"]: return
    x = np.load(m["npy"]).astype(np.float32) / 32767
    buf = io.BytesIO(); sf.write(buf, x, 16000, format="WAV", subtype="PCM_16")
    body = json.dumps({"content": {"parts": [{"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(buf.getvalue()).decode()}}]}}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request(BASE, data=body, headers={"Authorization": "Bearer " + TOKEN + ":fake-voice-check", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r: d = json.loads(r.read().decode())
            np.save(path, np.array(d["embedding"]["values"], dtype=np.float32))
            with lock:
                state["tokens"] += d.get("usageMetadata", {}).get("promptTokenCount", 0); state["done"] += 1
                high_rs = state["tokens"] / 1e6 * HIGH_USD_PER_MTOK * 100
                if high_rs > CAP_RS: state["stop"] = True
                if state["done"] % 100 == 0 or state["done"] == len(jobs):
                    log(f"embedded {state['done']}/{len(jobs)}; tokens {state['tokens']:,}; cost so far Rs {state['tokens']/1e6*0.20*100:.1f} (low price) to Rs {high_rs:.1f} (high price); failures {state['fail']}")
            return
        except Exception as e:
            time.sleep(2 ** attempt)
    with lock: state["fail"] += 1

log(f"{len(jobs)} clips to embed")
with ThreadPoolExecutor(6) as ex: list(ex.map(one, jobs))
log(f"finished: done {state['done']}, failures {state['fail']}, stopped for cost cap: {state['stop']}")
