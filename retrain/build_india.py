"""Real Indian-accented English read speech (Common Voice mirror), 6-second clips, group = speaker. Logs time and done/total."""
import json, random, time, subprocess, numpy as np, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
rnd=random.Random(8); man=[]; seen=set(json.load(open("india_used.json"))) if __import__("os").path.exists("india_used.json") else set()
def dec(b):
    raw=subprocess.run(["ffmpeg","-v","error","-i","pipe:0","-f","f32le","-ac","1","-ar","16000","-"],input=b,capture_output=True,check=True).stdout; return np.frombuffer(raw,dtype="float32").copy()
for sh in [1,2,3,4]:   # shard 0 was used for the diagnostic above; keep it out of training
    p=hf_hub_download("WillHeld/india_accent_cv",f"data/train-0000{sh}-of-00034.parquet",repo_type="dataset",local_dir="data/india")
    t=pq.read_table(p,columns=["client_id","audio"]).to_pandas(); idx=list(range(len(t))); rnd.shuffle(idx); k=0
    for j in idx:
        a=t.iloc[j]["audio"]; b=a["bytes"] if isinstance(a,dict) else a
        try: x=dec(b)
        except Exception: continue
        if len(x)<2*16000: continue
        x=x[:6*16000]; path=f"work/pool/india_{sh}_{j}.npy"; np.save(path,(x*32767).astype(np.int16))
        man.append(dict(dataset="IndianCV",label="real",generator="real",group="india:"+str(t.iloc[j]["client_id"])[:12],lang="en",npy=path,seconds=round(len(x)/16000,2))); k+=1
        if k>=170: break
    log(f"shard {sh}: {k} clips; total {len(man)}; speakers {len(set(m['group'] for m in man))}")
json.dump(man,open("work/india_manifest.json","w")); log("saved")
