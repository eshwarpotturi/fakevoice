"""Can embeddings fingerprint a voice? No training: real recordings only, compared by cosine similarity.

'Who is it?': enrol each person with 5 clips (their average), then name each other clip by the closest person.
'Same person?': random pairs of clips, scored by similarity; equal error rate is where wrong accepts equal wrong rejects.
Writes work/fingerprint.json. Every log line has a timestamp.
"""
import json, time
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
man = json.load(open("work/manifest.json"))
lab = np.array([m["label"] for m in man]); spk = np.array([m["speaker"] for m in man]); sp = np.array([m["split"] for m in man])
log("loading embeddings")
G = np.stack([np.load(f"work/gemini/clean_{m['file'].replace('.wav', '')}.npy") for m in man])
W = np.load("work/feat_wavlm.npy")[:, 3]
def eer(y, s):
    f, t, _ = roc_curve(y, s); i = np.nanargmin(np.abs((1 - t) - f)); return float((f[i] + 1 - t[i]) / 2)
def l2(X): return X / np.linalg.norm(X, axis=1, keepdims=True)
def verify(X, mask, n=20000, seed=0):
    idx = np.where(mask)[0]; rng = np.random.default_rng(seed); X = l2(X[idx]); s = spk[idx]
    a = rng.integers(0, len(idx), n); b = rng.integers(0, len(idx), n); k = a != b; a, b = a[k], b[k]
    y = (s[a] == s[b]).astype(int); sim = (X[a] * X[b]).sum(1)
    return float(roc_auc_score(y, sim)), eer(y, sim)
def identify(X, mask, enrol=5, seed=0):
    rng = np.random.default_rng(seed); idx = np.where(mask)[0]; X = l2(X); cent = {}; test = []
    for n in sorted(set(spk[idx])):
        ii = idx[spk[idx] == n]; rng.shuffle(ii)
        if len(ii) <= enrol + 3: continue
        c = X[ii[:enrol]].mean(0); cent[n] = c / np.linalg.norm(c); test += [(i, n) for i in ii[enrol:]]
    ns = list(cent); C = np.stack([cent[n] for n in ns])
    return float(np.mean([ns[int(np.argmax(C @ X[i]))] == n for i, n in test])), len(ns), len(test)
real = lab == "bona-fide"; out = []
for name, X in [("Gemini Embedding 2", G), ("WavLM, third layer", W)]:
    a12, n12, t12 = identify(X, real & (sp == "test")); a49, n49, t49 = identify(X, real)
    auc, e = verify(X, real)
    out.append({"name": name, "who12": a12, "people12": n12, "clips12": t12, "who49": a49, "people49": n49, "clips49": t49, "same_auc": auc, "same_eer": e})
    log(f"{name}: who-is-it {a12:.3f} ({n12} people), {a49:.3f} ({n49} people); same-person EER {e:.3f}")
json.dump(out, open("work/fingerprint.json", "w"), indent=1)
