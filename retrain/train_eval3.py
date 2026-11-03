"""Retrain the detector head with whole generators held out; compare layers, regularisation and epochs on the check split only.
Test split is used once at the end. Logs time and progress."""
import json, time, random, collections, numpy as np, pandas as pd, torch, librosa
from sklearn.metrics import roc_auc_score
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
torch.set_num_threads(2); rnd=random.Random(5)
# ---------- assemble rows
rows=[]; F={"clean":[], "phone":[]}
itw=json.load(open("work/itw_manifest.json")); fi=np.load("work/itw_feats.npz")
for k,m in enumerate(itw):
    rows.append(dict(family="InTheWild",dataset="InTheWild",y=int(m["label"]=="spoof"),generator="unknown",group=m["speaker"],split=m["split"],npy=m["npy"],lang="en"))
F["clean"].append(fi["clean"]); F["phone"].append(fi["phone"])
pool=json.load(open("work/pool_manifest.json")); fp=np.load("work/pool_feats.npz")
fam={"MLAAD":"MLAAD","WaveFake":"WaveFake","LJSpeech":"WaveFake","ASVspoof":"ASVspoof"}
for m in pool:
    rows.append(dict(family=fam[m["dataset"]],dataset=m["dataset"],y=int(m["label"]=="fake"),generator=m["generator"],group=m["group"],split=None,npy=m["npy"],lang=m["lang"]))
F["clean"].append(fp["clean"]); F["phone"].append(fp["phone"])
mic=json.load(open("work/mic_manifest.json")); fm_=np.load("work/mic_feats.npz")
for m in mic:
    rows.append(dict(family="MicSpeech",dataset=m["dataset"],y=0,generator="real",group=m["group"],split=("test_ami" if m["dataset"]=="AMI" else None),npy=m["npy"],lang="en"))
F["clean"].append(fm_["clean"]); F["phone"].append(fm_["phone"])
ind=json.load(open("work/india_manifest.json")); fi2=np.load("work/india_feats.npz")
for m in ind:
    rows.append(dict(family="IndianSpeech",dataset="IndianCV",y=0,generator="real",group=m["group"],split=None,npy=m["npy"],lang="en"))
F["clean"].append(fi2["clean"]); F["phone"].append(fi2["phone"])
d=pd.DataFrame(rows); X={k:np.concatenate(v).astype("float32") for k,v in F.items()}   # (N,13,768)
# ---------- split by whole generator (fakes) and by group (reals)
def assign(names, ft=0.2, fc=0.1):
    names=sorted(names); rnd.shuffle(names); n=len(names); nt=max(1,round(n*ft)); nc=max(1,round(n*fc))
    return {x:("test" if i<nt else "check" if i<nt+nc else "train") for i,x in enumerate(names)}
new=d.split.isna()
for fm in ["MLAAD","WaveFake","ASVspoof"]:
    f=d[new&(d.family==fm)&(d.y==1)]; gens=sorted(f.generator.unique())
    if fm=="WaveFake":                                  # Japanese folders have no real counterpart: test only
        ja=[g for g in gens if g.startswith("jsut")]; gens=[g for g in gens if g not in ja]
        d.loc[new&(d.generator.isin(ja)),"split"]="test_ja"
    mp=assign(gens); d.loc[new&(d.family==fm)&(d.y==1)&(d.generator.isin(mp)),"split"]=d.generator.map(mp)
    r=d[new&(d.family==fm)&(d.y==0)]
    if fm=="WaveFake": # single speaker: random split of clips
        idx=list(r.index); rnd.shuffle(idx); n=len(idx)
        for j,i in enumerate(idx): d.at[i,"split"]="test" if j<0.2*n else "check" if j<0.3*n else "train"
    else:
        mp=assign(r.group.unique()); d.loc[r.index,"split"]=r.group.map(mp)
for fm_name in ["MicSpeech","IndianSpeech"]:
    r=d[(d.family==fm_name)&(d.split.isna())]; mp=assign(r.group.unique()); d.loc[r.index,"split"]=r.group.map(mp)
assert d.split.notna().all()
# leakage checks: no generator or real group appears in two splits
assert d[d.family.isin(["MicSpeech","IndianSpeech"])].groupby("group").split.nunique().max()==1
for fm in ["MLAAD","WaveFake","ASVspoof"]:
    f=d[(d.family==fm)&(d.y==1)&(d.split!="test_ja")]; assert f.groupby("generator").split.nunique().max()==1
    if fm!="WaveFake": r=d[(d.family==fm)&(d.y==0)]; assert r.groupby("group").split.nunique().max()==1
log("split sizes:\n"+str(d.groupby(["family","split","y"]).size().unstack(fill_value=0)))
json.dump({fm:{s:sorted(d[(d.family==fm)&(d.y==1)&(d.split==s)].generator.unique()) for s in ["train","check","test","test_ja"]} for fm in ["MLAAD","WaveFake","ASVspoof"]},open("work/split_generators.json","w"),indent=1)
# ---------- loudness / tone only baseline features
def lt(p):
    x=np.load(p).astype("float32")/32767; return [20*np.log10(np.sqrt(np.mean(x**2))+1e-9), float(np.mean(librosa.feature.spectral_centroid(y=x,sr=16000)))]
d[["rms","cen"]]=[lt(p) for p in d.npy]; log("loudness features done")
# ---------- training helpers
tr=(d.split=="train").values; ck=(d.split=="check").values; te=(d.split=="test").values
w=1.0/d[tr].groupby(["family","y"]).y.transform("size").values; w=w/w.mean()
def fit(layer,wd,epochs,lr=0.01,record=None):
    Xtr=np.concatenate([X["clean"][tr,layer],X["phone"][tr,layer]]); ytr=np.concatenate([d.y[tr],d.y[tr]]); wtr=np.concatenate([w,w])
    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6; T=lambda a:torch.tensor((a-mu)/sd)
    xt,yt,wt=T(Xtr),torch.tensor(ytr,dtype=torch.float32),torch.tensor(wtr,dtype=torch.float32)
    lin=torch.nn.Linear(768,1); opt=torch.optim.AdamW(lin.parameters(),lr=lr,weight_decay=wd); curve=[]
    g=torch.Generator().manual_seed(0)
    for e in range(1,epochs+1):
        perm=torch.randperm(len(xt),generator=g)
        for s in range(0,len(xt),256):
            b=perm[s:s+256]; opt.zero_grad()
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(lin(xt[b])[:,0],yt[b],reduction="none")*wt[b]).mean(); loss.backward(); opt.step()
        if record is not None and e in record: curve.append((e,evalset(lin,mu,sd,layer,ck)))
    return lin,mu,sd,curve
def logits(lin,mu,sd,layer,mask,kind):
    with torch.no_grad(): return lin(torch.tensor((X[kind][mask,layer]-mu)/sd))[:,0].numpy()
def balanced_family_auc(mask,sc_by_kind):
    out=[]; sub=d[mask]; mic_=sub[sub.family.isin(["MicSpeech","IndianSpeech"])]
    for fm,g in sub.groupby("family"):
        if fm in ("MicSpeech","IndianSpeech") or g.y.nunique()<2: continue
        g=pd.concat([g,mic_])
        for kind in ("clean","phone"):
            s=pd.Series(sc_by_kind[kind],index=sub.index); out.append(roc_auc_score(g.y,s[g.index]))
    return float(np.mean(out))
def evalset(lin,mu,sd,layer,mask):
    return balanced_family_auc(mask,{k:logits(lin,mu,sd,layer,mask,k) for k in ("clean","phone")})
# ---------- choose layer, regularisation, epochs on the CHECK split
EP=[1,2,3,5,8,12,20]; res=[]
for layer in [3,5,6,7,8,9,10,12]:
    for wd in [0.0,0.1,1.0]:
        lin,mu,sd,curve=fit(layer,wd,20,record=EP)
        for e,a in curve: res.append(dict(layer=layer,wd=wd,epochs=e,check_auc=a))
        log(f"layer {layer} wd {wd}: best check AUC {max(a for _,a in curve):.3f}")
R=pd.DataFrame(res); R.to_csv("work/model_selection3.csv",index=False); best=R.sort_values("check_auc").iloc[-1]; log(f"chosen: {best.to_dict()}")
L,WD,E=int(best.layer),float(best.wd),int(best.epochs)
lin,mu,sd,_=fit(L,WD,E)
# Platt calibration on check clean+phone logits
from sklearn.linear_model import LogisticRegression
zc=np.concatenate([logits(lin,mu,sd,L,ck,"clean"),logits(lin,mu,sd,L,ck,"phone")]); yc=np.concatenate([d.y[ck],d.y[ck]])
pl=LogisticRegression(C=1e6).fit(zc[:,None],yc); coef,icpt=float(pl.coef_[0,0]),float(pl.intercept_[0])
np.savez("work/new3_head.npz",layer=L,mean=mu,scale=sd,coef=lin.weight.detach().numpy()[0],intercept=lin.bias.item(),platt_coef=coef,platt_icpt=icpt)
# ---------- loudness-only baseline
lt_m=LogisticRegression(class_weight="balanced").fit(d.loc[tr,["rms","cen"]],d.y[tr])
# ---------- TEST (used once)
tm=(d.split.isin(["test","test_ja","test_ami"])).values; d["new_clean"]=np.nan; d["new_phone"]=np.nan
for kind in ("clean","phone"):
    z=logits(lin,mu,sd,L,tm,kind); d.loc[tm,f"new_{kind}"]=1/(1+np.exp(-(coef*z+icpt)))
d.loc[tm,"loud"]=lt_m.predict_proba(d.loc[tm,["rms","cen"]])[:,1]
d.to_pickle("work/d3.pkl"); np.save("work/X3_clean.npy",X["clean"][:,L]); np.save("work/X3_phone.npy",X["phone"][:,L])
d.drop(columns=["npy"]).assign(npy=d.npy).to_csv("work/test_scores_new3.csv",index=False)
log("saved scores; run old-model scoring next")
