import sys; sys.path.insert(0, ".")   # run from the repository root
import json, numpy as np, pandas as pd, torch, onnxruntime as ort, soundfile as sf, librosa
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(2); pd.set_option("display.width",220)
d=pd.read_pickle("work/d3.pkl"); h=np.load("work/new3_head.npz"); h1=np.load("work/new_head.npz"); h2=np.load("work/new2_head.npz")
Xc=np.load("work/X3_clean.npy"); Xp=np.load("work/X3_phone.npy")
old=pd.read_csv("work/test_scores_all.csv")[["npy","old_clean","old_phone"]]
mo=pd.read_csv("work/mic_old_scores.csv")[["npy","old_clean","old_phone"]]
import os
if not os.path.exists("work/india_old_scores.csv"):
    from degrade import phone as _ph
    so_=ort.InferenceSession("docs/model/previous/detector.onnx",providers=["CPUExecutionProvider"]); oc_=json.load(open("docs/model/previous/config.json"))["platt"]
    pp=lambda x: float(min(.99,max(.05,1/(1+np.exp(-(oc_["coef"]*float(so_.run(None,{"audio":x[None].astype("float32")})[0][0])+oc_["intercept"]))))))
    il=pd.read_csv("work/india_test_list.csv"); rr=[]
    for pth in il.npy:
        x=np.load(pth).astype("float32")/32767; rr.append((pth,pp(x),pp(_ph(x))))
    pd.DataFrame(rr,columns=["npy","old_clean","old_phone"]).to_csv("work/india_old_scores.csv",index=False)
mo=pd.concat([mo,pd.read_csv("work/india_old_scores.csv")])
d=d.merge(pd.concat([old,mo]),on="npy",how="left")
# v1 (first retrain, no microphone speech) scores from its own head on layer-5 features
def v1(X): z=((X-h1["mean"])/h1["scale"])@h1["coef"]+float(h1["intercept"]); return 1/(1+np.exp(-(float(h1["platt_coef"])*z+float(h1["platt_icpt"]))))
d["v1_clean"]=v1(Xc); d["v1_phone"]=v1(Xp)
def v2(X): z=((X-h2["mean"])/h2["scale"])@h2["coef"]+float(h2["intercept"]); return 1/(1+np.exp(-(float(h2["platt_coef"])*z+float(h2["platt_icpt"]))))
d["v2_clean"]=v2(Xc); d["v2_phone"]=v2(Xp)
t=d[d.split=="test"]
def row(g,c):
    f=g[g.y==1][c]; r=g[g.y==0][c]
    return dict(auc=round(roc_auc_score(g.y,g[c]),3), fake_caught=round(100*(f>=.5).mean()), real_flagged=round(100*(r>=.5).mean()), n_fake=len(f), n_real=len(r))
rows=[]
for fam in ["InTheWild","MLAAD","WaveFake","ASVspoof"]:
    g=t[t.family==fam]
    for nm,c in [("current demo","old_clean"),("v2 (mic speech)","v2_clean"),("v3 (final: + Indian accents)","new_clean"),("v3, phone line","new_phone")]: rows.append(dict(family=fam,model=nm,**row(g,c)))
T=pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv("test_summary_v3.csv",index=False)
g=t[t.family!="InTheWild"]; print("\nPooled unseen generators:",len(g))
for nm,c in [("current demo","old_clean"),("v2","v2_clean"),("v3","new_clean"),("current demo phone","old_phone"),("v3 phone","new_phone")]: print(nm,row(g,c))
print("\nReal microphone speech flagged AI (>=50%): share of clips")
m=d[(d.family.isin(["MicSpeech","IndianSpeech"]))&(d.split.isin(["test","test_ami"]))]
M=m.groupby("dataset").agg(n=("y","size"),current_clean=("old_clean",lambda s:round(100*(s>=.5).mean())),v1_clean=("v1_clean",lambda s:round(100*(s>=.5).mean())),v2_clean=("v2_clean",lambda s:round(100*(s>=.5).mean())),v3_clean=("new_clean",lambda s:round(100*(s>=.5).mean())),current_phone=("old_phone",lambda s:round(100*(s>=.5).mean())),v3_phone=("new_phone",lambda s:round(100*(s>=.5).mean()))).reset_index()
print(M.to_string(index=False)); M.to_csv("real_speech_false_alarms.csv",index=False)
# AUC of each fake family against its own reals + all mic test reals (v2 vs current demo)
print("\nAUC with microphone reals added:")
mt=m
for fam in ["InTheWild","MLAAD","WaveFake","ASVspoof"]:
    g=pd.concat([t[t.family==fam],mt]); print(fam,"current",round(roc_auc_score(g.y,g.old_clean),3),"v3",round(roc_auc_score(g.y,g.new_clean),3),"v3 phone",round(roc_auc_score(g.y,g.new_phone),3))
pg=d[(d.split.isin(["test","test_ja"]))&(d.y==1)].groupby(["family","generator"]).agg(n=("y","size"),current_clean=("old_clean",lambda s:round(100*(s>=.5).mean())),v3_clean=("new_clean",lambda s:round(100*(s>=.5).mean())),current_phone=("old_phone",lambda s:round(100*(s>=.5).mean())),v3_phone=("new_phone",lambda s:round(100*(s>=.5).mean()))).reset_index()
pg.to_csv("held_out_generators_v3.csv",index=False); print("\nheld-out gens caught: median v3 clean",pg.v3_clean.median(),"min",pg.v3_clean.min()," current median",pg.current_clean.median())
print(pg[pg.v3_clean<70].to_string(index=False))
# reliability bins on all test (fake families + ITW), clean
tt=d[d.split=="test"]; edges=[0,.05,.2,.4,.6,.8,.95,1.0001]
for c in ["new_clean","new_phone"]:
    b=pd.cut(tt[c],edges,right=False); print(c); print(tt.groupby(b,observed=False).agg(n=("y","size"),obs=("y","mean"),pred=(c,"mean")).round(3).to_string())
# Eswar's own recording and clones, through the exported model
cfg=json.load(open("docs/model/config.json")); sig=lambda z:1/(1+np.exp(-z))
s=ort.InferenceSession("work/new3_detector.onnx",providers=["CPUExecutionProvider"]); from degrade import phone
def sc(x):
    W=96000; wins=[(a,a+W) for a in range(0,len(x)-W+1,W)]
    if len(x)-len(wins)*W>=32000 or not wins: wins.append((len(wins)*W,len(x)))
    z=np.mean([float(s.run(None,{"audio":x[a:b][None]})[0][0]) for a,b in wins[:12]]); return float(min(.99,max(.05,sig(float(h["platt_coef"])*z+float(h["platt_icpt"])))))
print("\nOwn recording and clones (v3):")
for n in ["eswar-real","eswar-clone-1","eswar-clone-2","eswar-clone-3"]:
    x,sr=sf.read(f"docs/data/samples/{n}.wav",dtype="float32",always_2d=True); x=x.mean(1); x=librosa.resample(x,orig_sr=sr,target_sr=16000) if sr!=16000 else x
    print(n,"clean",round(100*sc(x)),"% AI | phone",round(100*sc(phone(x))),"% AI")
d.to_pickle("work/d3_scored.pkl")
