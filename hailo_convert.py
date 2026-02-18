"""
Hailo DFC conversion: ONNX -> HAR (parse) -> HAR (quantized) -> HEF

This script must be run on a machine with the Hailo Dataflow Compiler installed
(typically x86 Linux; Docker images available from Hailo).
It does NOT run on the Raspberry Pi itself.

Steps:
  1. Parse the ONNX model into a Hailo Archive (HAR)
  2. Optimize / quantize to INT8 using calibration data
  3. Compile to HEF (Hailo Executable Format)
"""

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Convert ONNX model to Hailo HEF")
parser.add_argument("--tag", type=str, default="3ch_16-32-64",
                    help="Tag to identify model variant (default: 3ch_16-32-64)")
parser.add_argument("--data", type=str, default="data/QG_jets.npz",
                    help="Path to jet data (for calibration)")
parser.add_argument("--outdir", type=str, default="output",
                    help="Output directory (default: output)")
args = parser.parse_args()

TAG = args.tag
OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

# --- Stage 0: generate calibration data if needed ---

CALIB_FILE = os.path.join(OUTDIR, f"calib_data_{TAG}.npy")

try:
    calib_data = np.load(CALIB_FILE)
    print(f"Loaded calibration data: shape={calib_data.shape}")
except FileNotFoundError:
    print(f"Generating calibration data from {args.data} ...")
    from load_jets import load_jets, preprocess_all_jets
    X, y = load_jets(args.data)
    np.random.seed(42)
    idx = np.random.choice(len(X), size=1024, replace=False)
    calib_data = preprocess_all_jets(X[idx], R=0.4, npixels=32,
                                     pt_min=0.0, normalize=True)
    np.save(CALIB_FILE, calib_data)
    print(f"Saved calibration data to {CALIB_FILE}: shape={calib_data.shape}")

# --- Stage 1: Parse ONNX ---

from hailo_sdk_client import ClientRunner

onnx_path = os.path.join(OUTDIR, f"jet_classifier_{TAG}.onnx")
runner = ClientRunner(hw_arch="hailo8")
hn, npz = runner.translate_onnx_model(
    onnx_path,
    net_name="jet_classifier",
    start_node_names=["input"],
    end_node_names=["output"],
)
parsed_har = os.path.join(OUTDIR, f"jet_classifier_{TAG}_parsed.har")
runner.save_har(parsed_har)
print(f"Stage 1 complete: parsed HAR saved to {parsed_har}")

# --- Stage 2: Optimize (quantize to INT8) ---

# The calibration dict key must match Hailo's internal input layer name
# (shown in the parser log as: 'input': 'jet_classifier/input_layer1').
# Hailo expects NHWC (channels last), our data is NCHW (channels first) -> transpose
calib_data_nhwc = np.transpose(calib_data, (0, 2, 3, 1))  # (N,3,32,32) -> (N,32,32,3)
print(f"Calibration data transposed to NHWC: {calib_data_nhwc.shape}")
calib_dataset = {"jet_classifier/input_layer1": calib_data_nhwc}

# Load model script for better quantization settings
alls_path = os.path.join(OUTDIR, f"jet_classifier_{TAG}.alls")
if os.path.exists(alls_path):
    runner.load_model_script(alls_path)
    print(f"Loaded model script: {alls_path}")

runner.optimize(calib_dataset)
quantized_har = os.path.join(OUTDIR, f"jet_classifier_{TAG}_quantized.har")
runner.save_har(quantized_har)
print(f"Stage 2 complete: quantized HAR saved to {quantized_har}")

# --- Stage 3: Compile to HEF ---
# Note: This step requires a compatible libstdc++ (GLIBCXX_3.4.30+).
# If it fails on RHEL, use hailo_compile.py on Ubuntu 22.04 or in the Hailo Docker image.

try:
    hef = runner.compile()
    hef_path = os.path.join(OUTDIR, f"jet_classifier_{TAG}.hef")
    with open(hef_path, "wb") as f:
        f.write(hef)
    print(f"Stage 3 complete: HEF saved to {hef_path}")
except Exception as e:
    print(f"\nStage 3 (compile) failed: {e}")
    print(f"The quantized HAR file is ready: {quantized_har}")
    print("To compile it to HEF, run hailo_compile.py on Ubuntu 22.04 or in the Hailo Docker image.")
