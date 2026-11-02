"""Final comparison of detectors, with a phone-quality test.

Rules:
- Train on train speakers only.
- Choose layer and settings on the check speakers only.
- Report on the test speakers only (never used for any choice).

Writes work/results2.json and work/head.npz (the chosen browser model's head).
"""
import glob
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def tpr_at(y, s, t=0.05):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(t, fpr, tpr))


def ci(y, s, n=500):
    rng = np.random.default_rng(0)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def summary(y, s, thr):
    return {
        "auc": float(roc_auc_score(y, s)), "eer": eer(y, s),
        "accuracy": float(((s >= thr).astype(int) == y).mean()),
        "fakes_caught_at_5pct_false_alarms": tpr_at(y, s), "auc_95ci": ci(y, s),
    }


def per_speaker(y, s, spk):
    out = {}
    for sp in sorted(set(spk)):
        m = spk == sp
        if len(set(y[m])) == 2:
            out[sp] = {"auc": float(roc_auc_score(y[m], s[m])), "real": int((y[m] == 0).sum()), "fake": int((y[m] == 1).sum())}
    return out


def best_thr(y, s):
    c = np.unique(s)
    return float(c[int(np.argmax([((s >= t).astype(int) == y).mean() for t in c]))])


def main():
    man = json.load(open("work/manifest.json"))
    y = np.array([m["label"] == "spoof" for m in man]).astype(int)
    split = np.array([m["split"] for m in man])
    spk = np.array([m["speaker"] for m in man])
    tr, ck, te = split == "train", split == "check", split == "test"
    # phone manifests are in manifest order, filtered by split
    idx_ct = np.where(ck | te)[0]
    pm = json.load(open("work/manifest_phone.json"))
    assert [m["file"] for m in pm] == [man[i]["file"] for i in idx_ct]
    ck_p = ck[idx_ct]; te_p = te[idx_ct]
    y_p, spk_p = y[idx_ct], spk[idx_ct]
    out = {"counts": {"train": int(tr.sum()), "check": int(ck.sum()), "test": int(te.sum()),
                      "test_speakers": int(len(set(spk[te]))), "train_speakers": int(len(set(spk[tr]))),
                      "check_speakers": int(len(set(spk[ck])))}, "detectors": []}
    heads = {}

    def fit_eval(name, kind, Xc, Xp_ct, Xp_train=None, y_ptrain=None, select="clean", note="", key=None):
        """Xc: clean features (all clips). Xp_ct: phone features for check+test clips. Xp_train: phone features for train clips."""
        layers = Xc.shape[1]
        best = None
        for layer in range(layers):
            sc = StandardScaler().fit(Xc[tr][:, layer])
            Xt, yt = sc.transform(Xc[tr][:, layer]), y[tr]
            if Xp_train is not None:
                Xt = np.vstack([Xt, sc.transform(Xp_train[:, layer])]); yt = np.concatenate([yt, y_ptrain])
            for C in (0.01, 0.1, 1.0):
                clf = LogisticRegression(C=C, max_iter=3000).fit(Xt, yt)
                a_c = -log_loss(y[ck], clf.predict_proba(sc.transform(Xc[ck][:, layer]))[:, 1])
                a_p = -log_loss(y_p[ck_p], clf.predict_proba(sc.transform(Xp_ct[ck_p][:, layer]))[:, 1]) if Xp_ct is not None else a_c
                crit = a_c if select == "clean" else (a_c + a_p) / 2
                # prefer fewer layers (smaller browser model) unless clearly worse
                if best is None or crit > best["crit"]:
                    best = {"layer": layer, "C": C, "crit": crit, "check_clean": a_c, "check_phone": a_p}
        # smaller-model tie-break: lowest layer within 0.003 of the best criterion
        cands = []
        for layer in range(layers):
            sc = StandardScaler().fit(Xc[tr][:, layer])
            Xt, yt = sc.transform(Xc[tr][:, layer]), y[tr]
            if Xp_train is not None:
                Xt = np.vstack([Xt, sc.transform(Xp_train[:, layer])]); yt = np.concatenate([yt, y_ptrain])
            clf = LogisticRegression(C=best["C"], max_iter=3000).fit(Xt, yt)
            a_c = -log_loss(y[ck], clf.predict_proba(sc.transform(Xc[ck][:, layer]))[:, 1])
            a_p = -log_loss(y_p[ck_p], clf.predict_proba(sc.transform(Xp_ct[ck_p][:, layer]))[:, 1]) if Xp_ct is not None else a_c
            crit = a_c if select == "clean" else (a_c + a_p) / 2
            cands.append((layer, crit))
        top = max(c for _, c in cands)
        # smaller browser model: lowest layer whose check loss is within 5% of the best
        layer = min(l for l, c in cands if c >= top - 0.05 * abs(top))
        L = layer
        sc = StandardScaler().fit(Xc[tr][:, L])
        Xt, yt = sc.transform(Xc[tr][:, L]), y[tr]
        if Xp_train is not None:
            Xt = np.vstack([Xt, sc.transform(Xp_train[:, L])]); yt = np.concatenate([yt, y_ptrain])
        clf = LogisticRegression(C=best["C"], max_iter=3000).fit(Xt, yt)
        s_ck = clf.decision_function(sc.transform(Xc[ck][:, L]))
        s_ck_p = clf.decision_function(sc.transform(Xp_ct[ck_p][:, L])) if Xp_ct is not None else None
        thr = best_thr(y[ck], s_ck) if select == "clean" else best_thr(np.concatenate([y[ck], y_p[ck_p]]), np.concatenate([s_ck, s_ck_p]))
        s_te = clf.decision_function(sc.transform(Xc[te][:, L]))
        d = {"id": key or name, "name": name, "kind": kind, "layer": int(L), "C": best["C"], "note": note,
             "clean": summary(y[te], s_te, thr), "phone": None,
             "per_speaker_clean": per_speaker(y[te], s_te, spk[te]), "per_speaker_phone": None,
             "check_clean_auc": float(roc_auc_score(y[ck], s_ck))}
        if Xp_ct is not None:
            s_tp = clf.decision_function(sc.transform(Xp_ct[te_p][:, L]))
            d["phone"] = summary(y_p[te_p], s_tp, thr)
            d["per_speaker_phone"] = per_speaker(y_p[te_p], s_tp, spk_p[te_p])
        # Platt scaling on check scores (both conditions when used for selection)
        sc_all = s_ck if s_ck_p is None or select == "clean" else np.concatenate([s_ck, s_ck_p])
        y_all = y[ck] if s_ck_p is None or select == "clean" else np.concatenate([y[ck], y_p[ck_p]])
        pl = LogisticRegression().fit((sc_all - thr).reshape(-1, 1), y_all)
        d["platt"] = {"coef": float(pl.coef_[0, 0]), "intercept": float(pl.intercept_[0]), "threshold": thr}
        out["detectors"].append(d)
        heads[d["id"]] = {"mean": sc.mean_, "scale": sc.scale_, "coef": clf.coef_[0], "intercept": clf.intercept_[0], "layer": L, "thr": thr,
                          "platt_coef": d["platt"]["coef"], "platt_intercept": d["platt"]["intercept"]}
        log(f"{name}: layer {L}; test clean AUC {d['clean']['auc']:.3f} acc {d['clean']['accuracy']:.3f}" +
            (f"; phone AUC {d['phone']['auc']:.3f} acc {d['phone']['accuracy']:.3f}" if d["phone"] else ""))

    Xw = np.load("work/feat_wavlm.npy")
    Xwp = np.load("work/feat_wavlm_phone.npy") if os.path.exists("work/feat_wavlm_phone.npy") else None
    Xwpt = np.load("work/feat_wavlm_phonetrain.npy") if os.path.exists("work/feat_wavlm_phonetrain.npy") else None
    fit_eval("WavLM, trained on clean audio", "trained", Xw, Xwp, key="wavlm_clean",
             note="Trained only on clean recordings.")
    if Xwpt is not None:
        pmt = json.load(open("work/manifest_phone_train.json"))
        idx_tr = np.where(tr)[0]
        assert [m["file"] for m in pmt] == [man[i]["file"] for i in idx_tr]
        fit_eval("WavLM, trained on clean and phone-quality audio", "trained", Xw, Xwp, Xp_train=Xwpt, y_ptrain=y[idx_tr],
                 select="both", key="wavlm_both", note="Trained on clean recordings plus phone-quality copies of the same training clips.")
    if os.path.exists("work/feat_w2v2.npy"):
        Xvp = np.load("work/feat_w2v2_phone.npy") if os.path.exists("work/feat_w2v2_phone.npy") else None
        fit_eval("wav2vec 2.0, trained on clean audio", "trained", np.load("work/feat_w2v2.npy"), Xvp, key="w2v2_clean",
                 note="Trained only on clean recordings.")

    # Gemini Embedding 2 (embedded through the company's LLM Foundry gateway by gemini_embed.py)
    gdir = "work/gemini"
    if os.path.isdir(gdir) and len(os.listdir(gdir)) >= len(man) + len(pm):
        ld = lambda kind, m: np.load(f"{gdir}/{kind}_{m['file'].replace('.wav', '')}.npy")
        Xg = np.stack([ld("clean", m) for m in man])[:, None, :]
        Xgp = np.stack([ld("phone", m) for m in pm])[:, None, :]
        fit_eval("Gemini Embedding 2, trained on clean audio", "trained", Xg, Xgp, key="gemini_clean",
                 note="Gemini Embedding 2 turns each clip into 3,072 numbers; the same small classifier is trained on top, on clean recordings only.")

    # off-the-shelf detectors
    for f in sorted(glob.glob("work/offshelf_*.npy")):
        p = np.load(f)
        nm = f.split("offshelf_")[1][:-4].replace("__", "/")
        pf = f.replace("offshelf_", "offphone_")
        okc, okt = ck & ~np.isnan(p), te & ~np.isnan(p)
        d = {"id": "off:" + nm, "name": f"Ready-made detector ({nm.split('/')[-1]})", "kind": "off-the-shelf", "layer": None,
             "note": f"Downloaded from the Hugging Face model hub ({nm}). We did not train it.",
             "clean": summary(y[okt], p[okt], 0.5), "phone": None, "per_speaker_clean": per_speaker(y[okt], p[okt], spk[okt]),
             "per_speaker_phone": None, "check_clean_auc": float(roc_auc_score(y[okc], p[okc]))}
        if os.path.exists(pf):
            pp = np.load(pf)
            d["phone"] = summary(y_p[te_p], pp[te_p], 0.5)
            d["per_speaker_phone"] = per_speaker(y_p[te_p], pp[te_p], spk_p[te_p])
        out["detectors"].append(d)
        log(f"{d['name']}: clean AUC {d['clean']['auc']:.3f} acc {d['clean']['accuracy']:.3f}" + (f"; phone AUC {d['phone']['auc']:.3f}" if d["phone"] else ""))

    json.dump(out, open("work/results2.json", "w"), indent=1)
    np.savez("work/heads.npz", **{f"{k}__{kk}": np.asarray(vv) for k, h in heads.items() for kk, vv in h.items()})
    log("wrote work/results2.json and work/heads.npz")


if __name__ == "__main__":
    main()
