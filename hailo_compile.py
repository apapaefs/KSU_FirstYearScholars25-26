"""
Hailo DFC Stage 3 only: compile a quantized HAR to HEF.

Run this on a machine where the full Hailo DFC compile step works
(Ubuntu 22.04 recommended, or Hailo Docker image).

Usage:
    python hailo_compile.py [--tag 3ch_16-32-64]

If the compile fails on RHEL due to libstdc++ issues, use the Hailo Docker image:
    docker pull hailo.ai/hailo-ai-sw-suite:2024-10
    docker run --rm -v $(pwd):/workspace -w /workspace \\
        hailo-ai-sw-suite python hailo_compile.py
"""

import argparse
import os

parser = argparse.ArgumentParser(description="Compile quantized HAR to HEF")
parser.add_argument("--tag", type=str, required=True,
                    help="Tag identifying the model variant (e.g. 3ch_16-32-64)")
parser.add_argument("--outdir", type=str, default="output",
                    help="Output directory (default: output)")
parser.add_argument("--har", default=None, help="Quantized HAR file (overrides tag-based path)")
parser.add_argument("--hw-arch", default="hailo8", help="Target hardware: hailo8 or hailo8l")
args = parser.parse_args()

TAG = args.tag
OUTDIR = args.outdir

har_path = args.har if args.har else os.path.join(OUTDIR, f"jet_classifier_{TAG}_quantized.har")

from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch=args.hw_arch, har=har_path)
hef = runner.compile()
hef_path = os.path.join(OUTDIR, f"jet_classifier_{TAG}.hef")
with open(hef_path, "wb") as f:
    f.write(hef)
print(f"HEF saved to {hef_path}")
