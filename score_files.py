"""Score audio files the way the demo page does: 6-second slices, average, calibrated percentage."""
import json, sys, numpy as np, onnxruntime as ort, soundfile as sf, librosa
from degrade import phone
cfg = json.load(open("docs/model/config.json")); P = cfg["platt"]
sess = ort.InferenceSession("docs/model/detector.onnx", providers=["CPUExecutionProvider"])
sig = lambda z: 1 / (1 + np.exp(-z))
def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True); x = x.mean(1)
    return librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
def score(x):
    W = 6 * 16000; wins = [(s, s + W) for s in range(0, len(x) - W + 1, W)]
    rem = len(x) - len(wins) * W
    if rem >= 2 * 16000 or not wins: wins.append((len(wins) * W, len(x)))
    logits = [float(sess.run(None, {"audio": x[a:b][None]})[0][0]) for a, b in wins]
    p = lambda l: float(min(cfg["ceil"], max(cfg["floor"], sig(P["coef"] * (l - P["threshold"]) + P["intercept"]))))
    return p(np.mean(logits)), [round(p(l), 2) for l in logits]
for path in sys.argv[1:]:
    x = load(path)
    c, cw = score(x); h, hw = score(phone(x))
    print(f"{path.split('/')[-1][:34]:34s} {len(x)/16000:5.1f}s | clean: {c*100:3.0f}% AI {cw} | phone line: {h*100:3.0f}% AI {hw}")
