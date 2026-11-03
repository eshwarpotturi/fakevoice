import sys; sys.path.insert(0, ".")   # run from the repository root
"""Mean-pooled WavLM-base-plus hidden states (all 13 layers) for clean and phone-line versions of each clip.
Usage: feats.py manifest.json out.npz   Logs time and done/total."""
import sys, json, time, numpy as np, torch, collections
from transformers import AutoFeatureExtractor, AutoModel
from degrade import phone
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
torch.set_num_threads(2)
name="microsoft/wavlm-base-plus"; fe=AutoFeatureExtractor.from_pretrained(name); model=AutoModel.from_pretrained(name).eval()
norm=bool(fe.do_normalize); log(f"do_normalize={norm}")
man=json.load(open(sys.argv[1])); n=len(man)
def prep(x):
    x=x.astype("float32")
    return (x-x.mean())/np.sqrt(x.var()+1e-7) if norm else x
def run(batch):
    with torch.no_grad():
        hs=model(torch.tensor(np.stack(batch)),output_hidden_states=True).hidden_states
    return torch.stack([h.mean(1) for h in hs],1).numpy().astype("float16")   # (B,13,768)
audio=[np.load(m["npy"]).astype("float32")/32767 for m in man]
out={"clean":[None]*n,"phone":[None]*n}; done=0
for kind in ("clean","phone"):
    xs=audio if kind=="clean" else [phone(a) for a in audio]
    groups=collections.defaultdict(list)
    for i,x in enumerate(xs): groups[len(x)].append(i)
    for L,ids in groups.items():
        for s in range(0,len(ids),8):
            b=ids[s:s+8]; f=run([prep(xs[i]) for i in b])
            for k,i in enumerate(b): out[kind][i]=f[k]
            done+=len(b)
            if (done//8)%10==0: log(f"{kind}: {done}/{2*n}")
np.savez(sys.argv[2],clean=np.stack(out["clean"]),phone=np.stack(out["phone"])); log("done")
