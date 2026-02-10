"""
Hailo DFC Stage 3 only: compile a quantized HAR to HEF.

Run this on a machine where the full Hailo DFC compile step works
(Ubuntu 22.04 recommended, or Hailo Docker image).

Usage:
    python hailo_compile.py [--har jet_classifier_quantized.har]

If the compile fails on RHEL due to libstdc++ issues, use the Hailo Docker image:
    docker pull hailo.ai/hailo-ai-sw-suite:2024-10
    docker run --rm -v $(pwd):/workspace -w /workspace \\
        hailo-ai-sw-suite python hailo_compile.py
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--har", default="jet_classifier_quantized.har", help="Quantized HAR file")
parser.add_argument("--hw-arch", default="hailo8", help="Target hardware: hailo8 or hailo8l")
args = parser.parse_args()

from hailo_sdk_client import ClientRunner

runner = ClientRunner(hw_arch=args.hw_arch, har=args.har)
hef = runner.compile()
hef_path = "jet_classifier.hef"
with open(hef_path, "wb") as f:
    f.write(hef)
print(f"HEF saved to {hef_path}")
