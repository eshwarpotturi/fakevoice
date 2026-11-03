"""Build the extra training/test pool: 6-second 16 kHz clips with dataset, label, generator and group (speaker/book). Logs time and done/total."""
import json, random, time, os, io, collections, subprocess, concurrent.futures as cf
import numpy as np, soundfile as sf, librosa
from huggingface_hub import HfApi, hf_hub_download
from remotezip import RemoteZip
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
OUT="work/pool"; os.makedirs(OUT,exist_ok=True); rnd=random.Random(11); SR=16000; pool=[]
def load(p):
    try: x,sr=sf.read(p,dtype="float32",always_2d=True); x=x.mean(1)
    except Exception:
        raw=subprocess.run(["ffmpeg","-v","error","-i",p,"-f","f32le","-ac","1","-ar","16000","-"],capture_output=True,check=True).stdout; return np.frombuffer(raw,dtype="float32").copy()
    return librosa.resample(x,orig_sr=sr,target_sr=SR) if sr!=SR else x
def have(key,**meta):
    p=f"{OUT}/{key}.npy"
    if os.path.exists(p):
        try: return dict(meta,npy=p,seconds=round(len(np.load(p))/SR,2))
        except Exception: return None
    return None
def save(x,key,**meta):
    x=x[:6*SR]
    if len(x)<1.0*SR: return None
    p=f"{OUT}/{key}.npy"; np.save(p,(x*32767).astype(np.int16)); return dict(meta,npy=p,seconds=round(len(x)/SR,2))
# ---- MLAAD
R="mueller99/x"; R="mueller91/MLAAD-tiny"
files=[s.rfilename for s in HfApi().dataset_info(R,files_metadata=True).siblings if s.rfilename.endswith(".wav")]
fake=collections.defaultdict(list)
for f in files:
    p=f.split("/")
    if p[0]=="fake": fake[p[2]].append(f)
sel=[]
for m,fs in sorted(fake.items()): rnd.shuffle(fs); sel+=[("fake",m,f) for f in fs[:10]]
real=[f for f in files if f.startswith("original/")]; rnd.shuffle(real)
sel+=[("real","M-AILABS",f) for f in real[:500]]
log(f"MLAAD selected {len(sel)}")
def g(s): return s, hf_hub_download(R,s[2],repo_type="dataset",local_dir="data/mlaad")
with cf.ThreadPoolExecutor(8) as ex:
    for i,(s,path) in enumerate(ex.map(g,sel),1):
        base=os.path.basename(s[2])[:-4]; grp="_".join(base.split("_")[:-3]) or base
        r=have(f"mlaad_{i}",dataset="MLAAD",label=s[0],generator=s[1] if s[0]=="fake" else "real",group=s[1] if s[0]=="fake" else "book:"+grp,lang=s[2].split("/")[1]) or save(load(path),f"mlaad_{i}",dataset="MLAAD",label=s[0],generator=s[1] if s[0]=="fake" else "real",group=s[1] if s[0]=="fake" else "book:"+grp,lang=s[2].split("/")[1])
        if r: pool.append(r)
        if i%100==0: log(f"MLAAD {i}/{len(sel)}")
# ---- WaveFake (fakes only, 20 per folder)
names=[n for n in json.load(open("data/wavefake_names.json")) if n.endswith(".wav")]; by={}
for n in names: by.setdefault(n.split("/")[-2],[]).append(n)
sel=[]
for k,fs in sorted(by.items()): rnd.shuffle(fs); sel+=[(k,f) for f in fs[:20]]
z=RemoteZip("https://zenodo.org/api/records/5642694/files/generated_audio.zip/content")
for i,(k,f) in enumerate(sel,1):
    m=dict(dataset="WaveFake",label="fake",generator=k,group=k,lang="ja" if k.startswith("jsut") else "en"); r=have(f"wf_{i}",**m)
    if r is None: z.extract(f,"data/wavefake"); r=save(load(os.path.join("data/wavefake",f)),f"wf_{i}",**m)
    if r: pool.append(r)
    if i%50==0: log(f"WaveFake {i}/{len(sel)}")
# ---- LJSpeech reals (source of WaveFake fakes)
import pyarrow.parquet as pq
p=hf_hub_download("MikhailT/lj-speech","data/full-00000-of-00008-0cc4e31b6b21f574.parquet",repo_type="dataset",local_dir="data/lj")
t=pq.read_table(p).to_pandas(); log(f"LJ shard rows {len(t)} cols {list(t.columns)}")
acol=[c for c in t.columns if "audio" in c.lower()][0]; idx=list(range(len(t))); rnd.shuffle(idx)
n=0
for j in idx[:230]:
    b=t.iloc[j][acol]; b=b["bytes"] if isinstance(b,dict) else b
    x,sr=sf.read(io.BytesIO(b),dtype="float32",always_2d=True); x=x.mean(1); x=librosa.resample(x,orig_sr=sr,target_sr=SR) if sr!=SR else x
    r=have(f"lj_{j}",dataset="LJSpeech",label="real",generator="real",group="ljspeech",lang="en") or save(x,f"lj_{j}",dataset="LJSpeech",label="real",generator="real",group="ljspeech",lang="en")
    if r: pool.append(r); n+=1
log(f"LJSpeech {n} real")
# ---- ASVspoof 2021 DF (no codec): 15 fakes per 2019 attack, 200 real
names=json.load(open("data/asv_names.json")); idx2={os.path.basename(n)[:-5]:n for n in names if n.endswith(".flac")}
fk=collections.defaultdict(list); rl=[]
for l in open("data/keys/DF/CM/trial_metadata.txt"):
    q=l.split()
    if q[2]!="nocodec" or q[1] not in idx2: continue
    if q[5]=="bonafide": rl.append((q[1],q[3],q[0]))
    elif q[3]=="asvspoof": fk[q[4]].append((q[1],q[0]))
sel=[]
for a,ids in sorted(fk.items()): rnd.shuffle(ids); sel+=[("fake",a,i,spk) for i,spk in ids[:15]]
rnd.shuffle(rl); sel+=[("real","real",i,src+":"+spk) for i,src,spk in rl[:200]]
z=RemoteZip("https://huggingface.co/datasets/Bisher/ASVspoof_2021_DF/resolve/main/ASVspoof_DF_2021.zip")
for i,(lab,gen,fid,grp) in enumerate(sel,1):
    m=dict(dataset="ASVspoof",label=lab,generator=gen,group=grp if lab=="real" else gen,lang="en"); r=have(f"asv_{i}",**m)
    if r is None:
        try: z.extract(idx2[fid],"data/asv"); r=save(load(os.path.join("data/asv",idx2[fid])),f"asv_{i}",**m)
        except Exception as e: log(f"skip {fid}: {e}"); continue
    if r: pool.append(r)
    if i%50==0: log(f"ASVspoof {i}/{len(sel)}")
json.dump(pool,open("work/pool_manifest.json","w")); log(f"saved {len(pool)} clips"); log(str(collections.Counter((r['dataset'],r['label']) for r in pool)))
