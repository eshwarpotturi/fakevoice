"""Export the chosen detector as one ONNX file: waveform in, score out.

Usage: python3 export_onnx.py <layer> <model_dir_out>
Reads work/head.npz (scaler mean/scale, coef, intercept) when it exists.
"""
import sys, time, json
import numpy as np, torch
from transformers import AutoFeatureExtractor, AutoModel

def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

class Detector(torch.nn.Module):
    def __init__(self, base, layer, mean, scale, coef, intercept, normalise):
        super().__init__()
        self.base, self.layer, self.normalise = base, layer, normalise
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("coef", torch.tensor(coef, dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(intercept, dtype=torch.float32))
    def forward(self, audio):
        x = audio
        if self.normalise:
            x = (x - x.mean(dim=1, keepdim=True)) / torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-7)
        hs = self.base(x, output_hidden_states=True).hidden_states[self.layer]
        v = hs.mean(dim=1)[0]
        z = ((v - self.mean) / self.scale * self.coef).sum() + self.intercept
        return z.reshape(1)

def build(layer, head):
    name = "microsoft/wavlm-base-plus"
    fe = AutoFeatureExtractor.from_pretrained(name)
    base = AutoModel.from_pretrained(name).eval()
    base.encoder.layers = base.encoder.layers[:layer]       # drop the layers we do not use
    d = Detector(base, layer, head["mean"], head["scale"], head["coef"], float(head["intercept"]), bool(fe.do_normalize)).eval()
    return d, fe

if __name__ == "__main__":
    layer = int(sys.argv[1]); out = sys.argv[2]
    head = np.load("work/head.npz") if len(sys.argv) < 4 else None
    if head is None:
        head = {"mean": np.zeros(768), "scale": np.ones(768), "coef": np.ones(768) * 0.01, "intercept": 0.0}
    d, fe = build(layer, head)
    log(f"do_normalize={fe.do_normalize}")
    x = torch.randn(1, 16000 * 6)
    with torch.no_grad(): ref = d(x)
    torch.onnx.export(d, (x,), out, input_names=["audio"], output_names=["logit"],
                      dynamic_axes={"audio": {1: "n"}}, opset_version=17, dynamo=False)
    log(f"exported {out}")
