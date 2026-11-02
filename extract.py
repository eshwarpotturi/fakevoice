"""Extract features and off-the-shelf detector scores for every pulled clip.

Outputs (in work/):
  feat_<model>.npy      layer-wise mean-pooled embeddings, shape (clips, layers, dim)
  offshelf_<name>.npy   probability that each clip is AI-generated
Every log line has a timestamp and done/total progress.
"""
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioClassification

torch.set_num_threads(8)
SR = 16000


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load(m):
    x = np.load(m["npy"]).astype(np.float32) / 32767.0
    return x


def embed(model_name, manifest):
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    out = []
    t0 = time.time()
    for i, m in enumerate(manifest, 1):
        x = load(m)
        inp = fe(x, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            hs = model(**inp, output_hidden_states=True).hidden_states
        out.append(torch.stack([h.mean(dim=1).squeeze(0) for h in hs]).numpy())
        if i % 100 == 0 or i == len(manifest):
            log(f"{model_name}: {i}/{len(manifest)} ({time.time() - t0:.0f}s)")
    return np.stack(out)


def offshelf(model_name, manifest, fake_words=("fake", "spoof", "ai", "synthetic", "deepfake")):
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name).eval()
    labels = {int(k): v for k, v in model.config.id2label.items()}
    log(f"{model_name} labels: {labels}")
    fake_ids = [k for k, v in labels.items() if any(w == v.lower() or w in v.lower() for w in fake_words)
                and "real" not in v.lower() and "bona" not in v.lower()]
    log(f"{model_name} treated as fake: {fake_ids}")
    probs = []
    t0 = time.time()
    for i, m in enumerate(manifest, 1):
        x = load(m)
        inp = fe(x, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(model(**inp).logits, dim=-1).squeeze(0).numpy()
        probs.append(float(p[fake_ids].sum()))
        if i % 100 == 0 or i == len(manifest):
            log(f"{model_name}: {i}/{len(manifest)} ({time.time() - t0:.0f}s)")
    return np.array(probs)


def main():
    manifest = json.load(open("work/manifest.json"))
    which = sys.argv[1:]
    for w in which:
        if w == "wavlm":
            np.save("work/feat_wavlm.npy", embed("microsoft/wavlm-base-plus", manifest))
        elif w == "w2v2":
            np.save("work/feat_w2v2.npy", embed("facebook/wav2vec2-base", manifest))
        elif w in ("wavlm_phone", "w2v2_phone"):
            pm = json.load(open("work/manifest_phone.json"))
            name = "microsoft/wavlm-base-plus" if w == "wavlm_phone" else "facebook/wav2vec2-base"
            np.save(f"work/feat_{w}.npy", embed(name, pm))
        elif w == "wavlm_phonetrain":
            pm = json.load(open("work/manifest_phone_train.json"))
            np.save("work/feat_wavlm_phonetrain.npy", embed("microsoft/wavlm-base-plus", pm))
        elif w.startswith("offphone:"):
            name = w[9:]
            pm = json.load(open("work/manifest_phone.json"))
            p = offshelf(name, pm)
            np.save(f"work/offphone_{name.replace('/', '__')}.npy", p)
        elif w.startswith("off:"):
            name = w[4:]
            sub = [m for m in manifest if m["split"] in ("check", "test")]
            idx = [i for i, m in enumerate(manifest) if m["split"] in ("check", "test")]
            p = offshelf(name, sub)
            full = np.full(len(manifest), np.nan)
            full[idx] = p
            np.save(f"work/offshelf_{name.replace('/', '__')}.npy", full)
    log("done")


if __name__ == "__main__":
    main()
