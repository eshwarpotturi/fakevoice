"""Shortcut check: can a trivial loudness/bandwidth summary separate real from fake?"""
import json, time
import numpy as np, librosa
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
man = json.load(open("work/manifest.json"))
def feats(path):
    x = np.load(path).astype(np.float32)/32767
    S = np.abs(librosa.stft(x, n_fft=512, hop_length=256))**2
    ls = np.log10(S.mean(axis=1)+1e-10)            # 257-bin average log spectrum
    bands = ls.reshape(-1)[:256].reshape(16,16).mean(axis=1)   # 16 coarse bands
    rms = np.log10(np.sqrt((x**2).mean())+1e-9)
    return np.concatenate([bands,[rms]])
X = []
for i,m in enumerate(man,1):
    X.append(feats(m["npy"]))
    if i%500==0 or i==len(man): log(f"{i}/{len(man)}")
X = np.array(X); np.save("work/feat_simple.npy", X)
y = np.array([m["label"]=="spoof" for m in man]).astype(int)
sp = np.array([m["split"] for m in man])
sc = StandardScaler().fit(X[sp=="train"])
clf = LogisticRegression(C=1,max_iter=2000).fit(sc.transform(X[sp=="train"]), y[sp=="train"])
res = {}
for s in ("check","test"):
    m = sp==s
    res[s] = float(roc_auc_score(y[m], clf.decision_function(sc.transform(X[m]))))
    log(f"simple loudness+bandwidth model, unseen speakers, {s}: AUC {res[s]:.3f}")
json.dump(res, open("work/shortcut.json","w"))
