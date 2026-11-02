"""Fit the percentage so it means what it says.

Out-of-fold scores: for each fold, train on other training people, score the held-out people
(clean copy and phone copy of each clip). Plus the check people scored by the final model.
The test people are not used.
Writes platt values back into work/heads.npz and work/calibration.json.
"""
import json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
L = 3
man = json.load(open("work/manifest.json"))
y = np.array([m["label"] == "spoof" for m in man]).astype(int); sp = np.array([m["split"] for m in man]); spk = np.array([m["speaker"] for m in man])
tr, ck, te = sp == "train", sp == "check", sp == "test"
Fc = np.load("work/feat_wavlm.npy")[:, L, :]; Fpt = np.load("work/feat_wavlm_phonetrain.npy")[:, L, :]    # phone copies of train clips (train order)
Fp_ct = np.load("work/feat_wavlm_phone.npy")[:, L, :]
idx_ct = np.where(ck | te)[0]; idx_tr = np.where(tr)[0]
h = np.load("work/heads.npz"); k = "wavlm_both__"; C = 0.1
oof_s, oof_y = [], []
for f, (a, b) in enumerate(GroupKFold(5).split(idx_tr, groups=spk[idx_tr]), 1):
    Xtr = np.vstack([Fc[idx_tr][a], Fpt[a]]); ytr = np.concatenate([y[idx_tr][a]] * 2)
    sc = StandardScaler().fit(Xtr); clf = LogisticRegression(C=float(h[k + "C"]) if k + "C" in h.files else C, max_iter=3000).fit(sc.transform(Xtr), ytr)
    for X in (Fc[idx_tr][b], Fpt[b]):
        oof_s.append(clf.decision_function(sc.transform(X))); oof_y.append(y[idx_tr][b])
    log(f"fold {f}/5 done")
# check people, scored by the final model
mean, scale, coef, ic = h[k + "mean"], h[k + "scale"], h[k + "coef"], float(h[k + "intercept"])
score = lambda X: ((X - mean) / scale * coef).sum(1) + ic
ck_p = ck[idx_ct]
oof_s += [score(Fc[ck]), score(Fp_ct[ck_p])]; oof_y += [y[ck], y[idx_ct][ck_p]]
s = np.concatenate(oof_s); yy = np.concatenate(oof_y)
pl = LogisticRegression(C=1e6).fit(s.reshape(-1, 1), yy)      # almost no regularisation, plain Platt
a, b = float(pl.coef_[0, 0]), float(pl.intercept_[0])
log(f"platt on {len(s)} out-of-fold scores: coef {a:.3f}, intercept {b:.3f}")
# evaluate on test (clean + phone)
te_p = te[idx_ct]
st = np.concatenate([score(Fc[te]), score(Fp_ct[te_p])]); yt = np.concatenate([y[te], y[idx_ct][te_p]])
pt = 1 / (1 + np.exp(-(a * st + b)))
log(f"test log loss {log_loss(yt, pt):.3f}")
bins = [0, .05, .2, .4, .6, .8, .95, 1.0001]; rows = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (pt >= lo) & (pt < hi)
    if m.sum():
        rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "observed_fake": float(yt[m].mean()), "predicted": float(pt[m].mean())})
        log(f"predicted {lo:.2f}-{min(hi,1):.2f}: n={m.sum():4d} observed fake share {yt[m].mean():.2f} mean predicted {pt[m].mean():.2f}")
s0 = -b / a    # the point where the percentage is 50%
acc_clean = float(((score(Fc[te]) >= s0).astype(int) == y[te]).mean())
acc_phone = float(((score(Fp_ct[te_p]) >= s0).astype(int) == y[idx_ct][te_p]).mean())
log(f"at the demo's 50% point: test accuracy clean {acc_clean:.3f}, phone {acc_phone:.3f}")
json.dump({"coef": a, "intercept": b, "acc_clean": acc_clean, "acc_phone": acc_phone, "test_bins": rows, "test_log_loss": float(log_loss(yt, pt))}, open("work/calibration.json", "w"), indent=1)
d = dict(np.load("work/heads.npz")); d[k + "platt_coef"] = np.array(a); d[k + "platt_intercept"] = np.array(b); d[k + "thr"] = np.array(0.0)
np.savez("work/heads.npz", **d)
