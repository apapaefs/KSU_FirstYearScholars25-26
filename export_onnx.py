import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from model import JetClassifierCNN

parser = argparse.ArgumentParser(description="Export jet classifier to ONNX")
parser.add_argument("--tag", type=str, required=True,
                    help="Tag identifying the model variant (e.g. 3ch_16-32-64)")
parser.add_argument("--outdir", type=str, default="output",
                    help="Output directory (default: output)")
args = parser.parse_args()

TAG = args.tag
OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

# wrapper that includes sigmoid in the exported model
class JetClassifierExport(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.base(x))


# load trained model (checkpoint includes architecture params)
model_path = os.path.join(OUTDIR, f"best_jet_classifier_{TAG}.pt")
checkpoint = torch.load(model_path, map_location="cpu")
model = JetClassifierCNN(
    c1=checkpoint['c1'], c2=checkpoint['c2'], c3=checkpoint['c3'],
    c4=checkpoint.get('c4'), fc=checkpoint['fc'],
)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
print(f"Loaded model: c1={checkpoint['c1']}, c2={checkpoint['c2']}, "
      f"c3={checkpoint['c3']}, c4={checkpoint.get('c4')}, fc={checkpoint['fc']}")

export_model = JetClassifierExport(model)
export_model.eval()

# dummy input (batch=1, channels=3, height=32, width=32)
dummy_input = torch.randn(1, 3, 32, 32)

# export to ONNX using the legacy TorchScript-based exporter
# (the new torch.export-based exporter produces graphs that Hailo DFC can't parse)
onnx_path = os.path.join(OUTDIR, f"jet_classifier_{TAG}.onnx")
torch.onnx.export(
    export_model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes=None,   # fixed shapes for Hailo
    dynamo=False,         # force legacy TorchScript exporter
)
print(f"Exported to {onnx_path}")

# validate the ONNX model
import onnx
model_onnx = onnx.load(onnx_path)
onnx.checker.check_model(model_onnx)
print("ONNX model validation passed.")

# test with ONNX Runtime
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path)
result = sess.run(None, {"input": dummy_input.numpy()})
print(f"ONNX Runtime test output: {result[0]}")

# compare against PyTorch
with torch.no_grad():
    pt_result = export_model(dummy_input).numpy()
print(f"PyTorch output:           {pt_result}")
print(f"Max difference:           {np.abs(result[0] - pt_result).max():.2e}")
