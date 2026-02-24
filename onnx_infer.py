"""
ONNX Runtime inference script for Raspberry Pi (or any machine).
No Hailo hardware required — runs on CPU using the ONNX model directly.

Usage:
    python onnx_infer.py [--tag 3ch_16-32-64] [--data data/QG_jets.npz] [--n 100]

Requirements:
    pip install numpy onnxruntime
"""

import argparse
import time
import os
import numpy as np
import onnxruntime as ort
from load_jets import load_jets, jet_to_image_3ch


def run_inference(onnx_path, jet_images):
    """
    Run inference using ONNX Runtime.
    jet_images: numpy array of shape (N, 3, 32, 32), float32
    Returns: numpy array of shape (N,) with probabilities
    """
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    results = []
    for i in range(len(jet_images)):
        inp = jet_images[i:i+1]  # (1, 3, 32, 32)
        out = sess.run(None, {input_name: inp})[0]
        results.append(out.flatten()[0])

    return np.array(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run jet classifier with ONNX Runtime")
    parser.add_argument("--tag", type=str, required=True,
                        help="Tag identifying the model variant (e.g. 3ch_16-32-64)")
    parser.add_argument("--outdir", type=str, default="output",
                        help="Output directory where ONNX lives (default: output)")
    parser.add_argument("--model", default=None, help="Path to ONNX model (overrides tag-based path)")
    parser.add_argument("--data", default="data/QG_jets.npz", help="Path to jet data")
    parser.add_argument("--n", type=int, default=100, help="Number of jets to classify")
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(args.outdir, f"jet_classifier_{args.tag}.onnx")

    # load jets
    X, y = load_jets(args.data)
    n = min(args.n, len(X))

    # convert to 3-channel images
    print(f"Preprocessing {n} jets...")
    t0 = time.time()
    test_images = np.stack([
        jet_to_image_3ch(X[i], R=0.4, npixels=32, pt_min=0.0, normalize=True)
        for i in range(n)
    ])
    print(f"  Preprocessing done in {time.time()-t0:.2f}s")

    # run inference
    print(f"Running ONNX Runtime inference on {n} jets...")
    t0 = time.time()
    probs = run_inference(args.model, test_images)
    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.2f}s ({elapsed/n*1000:.1f} ms/jet)")

    # print first 20 results
    print(f"\nFirst 20 results:")
    for i in range(min(20, n)):
        label = "quark" if y[i] == 1 else "gluon"
        pred  = "quark" if probs[i] > 0.5 else "gluon"
        mark  = "ok" if label == pred else "WRONG"
        print(f"  Jet {i:3d}: true={label:5s}  pred={pred:5s}  prob={probs[i]:.3f}  {mark}")

    # overall accuracy
    preds = (probs > 0.5).astype(int)
    accuracy = (preds == y[:n]).mean()
    print(f"\nOverall accuracy on {n} jets: {accuracy:.4f}")
