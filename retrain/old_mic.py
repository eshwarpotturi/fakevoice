import sys; sys.path.insert(0, ".")   # run from the repository root
import json, time, numpy as np, pandas as pd, onnxruntime as ort
from degrade import phone
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)
cfg=json.load(open("docs/model/previous/config.json")); P=cfg["platt"]; so=ort.SessionOptions(); so.intra_op_num_threads=2
sess=ort.InferenceSession("docs/model/previous/detector.onnx",so,providers=["CPUExecutionProvider"]); sig=lambda z:1/(1+np.exp(-z))
d=pd.read_csv("work/mic_test_list.csv"); n=len(d)
def p(x): return float(min(cfg["ceil"],max(cfg["floor"],sig(P["coef"]*(float(sess.run(None,{"audio":x[None].astype("float32")})[0][0])-P["threshold"])+P["intercept"]))))
oc,op=[],[]
for i,path in enumerate(d.npy,1):
    x=np.load(path).astype("float32")/32767; oc.append(p(x)); op.append(p(phone(x)))
    if i%50==0: log(f"scored {i}/{n}")
d["old_clean"]=oc; d["old_phone"]=op; d.to_csv("work/mic_old_scores.csv",index=False); log("done")
