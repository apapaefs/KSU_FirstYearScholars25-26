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

# --- Stage 0: generate calibration data if needed ---

CALIB_FILE = "calib_data.npy"

try:
    calib_data = np.load(CALIB_FILE)
    print(f"Loaded calibration data: shape={calib_data.shape}")
except FileNotFoundError:
    print("Generating calibration data from QG_jets.npz ...")
    from load_jets import load_jets, preprocess_all_jets
    X, y = load_jets("QG_jets.npz")
    np.random.seed(42)
    idx = np.random.choice(len(X), size=1024, replace=False)
    calib_data = preprocess_all_jets(X[idx], R=0.4, npixels=32,
                                     pt_min=0.0, normalize=True)
    np.save(CALIB_FILE, calib_data)
    print(f"Saved calibration data to {CALIB_FILE}: shape={calib_data.shape}")

# --- Stage 1: Parse ONNX ---

from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch="hailo8")
hn, npz = runner.translate_onnx_model(
    "jet_classifier.onnx",
    net_name="jet_classifier",
    start_node_names=["input"],
    end_node_names=["output"],
)
runner.save_har("jet_classifier_parsed.har")
print("Stage 1 complete: parsed HAR saved.")

# --- Stage 2: Optimize (quantize to INT8) ---

# The calibration dict key must match Hailo's internal input layer name
# (shown in the parser log as: 'input': 'jet_classifier/input_layer1').
# Hailo expects NHWC (channels last), our data is NCHW (channels first) -> transpose
calib_data_nhwc = np.transpose(calib_data, (0, 2, 3, 1))  # (N,3,32,32) -> (N,32,32,3)
print(f"Calibration data transposed to NHWC: {calib_data_nhwc.shape}")
calib_dataset = {"jet_classifier/input_layer1": calib_data_nhwc}

# Load model script for better quantization settings
import os
alls_path = "jet_classifier.alls"
if os.path.exists(alls_path):
    runner.load_model_script(alls_path)
    print(f"Loaded model script: {alls_path}")

# Try to force optimization level 2 even on CPU (may be slow but more accurate)
try:
    runner.optimize(calib_dataset, optimization_level=2)
except TypeError:
    # Older API may not accept optimization_level kwarg
    runner.optimize(calib_dataset)
runner.save_har("jet_classifier_quantized.har")
print("Stage 2 complete: quantized HAR saved.")

# --- Stage 3: Compile to HEF ---
# Note: This step requires a compatible libstdc++ (GLIBCXX_3.4.30+).
# If it fails on RHEL, use hailo_compile.py on Ubuntu 22.04 or in the Hailo Docker image.

try:
    hef = runner.compile()
    hef_path = "jet_classifier.hef"
    with open(hef_path, "wb") as f:
        f.write(hef)
    print(f"Stage 3 complete: HEF saved to {hef_path}")
except Exception as e:
    print(f"\nStage 3 (compile) failed: {e}")
    print("The quantized HAR file is ready: jet_classifier_quantized.har")
    print("To compile it to HEF, run hailo_compile.py on Ubuntu 22.04 or in the Hailo Docker image.")
