"""Real speech recorded on ordinary microphones: LibriSpeech (other), FLEURS (en_us), AMI (headset, conversational).
6-second 16 kHz clips; group = speaker where known. Logs time and done/total."""
import io, json, random, time, subprocess, os, numpy as np, soundfile as sf, librosa, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
OUT="work/pool"; SR=16000; rnd=random.Random(21); man=[]
def dec(b):
    try: x,sr=sf.read(io.BytesIO(b),dtype="float32",always_2d=True); x=x.mean(1)
    except Exception:
        raw=subprocess.run(["ffmpeg","-v","error","-i","pipe:0","-f","f32le","-ac","1","-ar","16000","-"],input=b,capture_output=True,check=True).stdout; return np.frombuffer(raw,dtype="float32").copy()
    return librosa.resample(x,orig_sr=sr,target_sr=SR) if sr!=SR else x
def take(repo,fn,name,n,speaker_col,min_s):
    p=hf_hub_download(repo,fn,repo_type="dataset",local_dir="data/mic"); t=pq.read_table(p).to_pandas(); log(f"{name}: {len(t)} rows, cols {list(t.columns)}")
    idx=list(range(len(t))); rnd.shuffle(idx); k=0
    for j in idx:
        a=t.iloc[j]["audio"]; b=a["bytes"] if isinstance(a,dict) else a
        x=dec(b)
        if len(x)<min_s*SR: continue
        x=x[:6*SR]; path=f"{OUT}/mic_{name}_{j}.npy"; np.save(path,(x*32767).astype(np.int16))
        g=str(t.iloc[j][speaker_col]) if speaker_col else "id:"+str(t.iloc[j].get("id",j))
        man.append(dict(dataset=name,label="real",generator="real",group=f"{name}:{g}",lang="en",npy=path,seconds=round(len(x)/SR,2))); k+=1
        if k%50==0: log(f"{name} {k}/{n}")
        if k>=n: break
take("openslr/librispeech_asr","all/test.other/0000.parquet","LibriSpeech",250,"speaker_id",2.0)
take("google/fleurs","parquet-data/en_us/test-00000-of-00001.parquet","FLEURS",250,None,2.0)
take("edinburghcstr/ami","ihm/test-00000-of-00004.parquet","AMI",250,"speaker_id",2.0)
json.dump(man,open("work/mic_manifest.json","w")); log(f"saved {len(man)}")
