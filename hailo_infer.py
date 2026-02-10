"""
HailoRT inference script for Raspberry Pi 5 with Hailo-8 AI HAT+.

Usage:
    python hailo_infer.py [--hef jet_classifier.hef] [--data QG_jets.npz] [--n 100]
"""

import argparse
import numpy as np
import hailo_platform as hpf
from load_jets import load_jets, jet_to_image_3ch


def run_inference(hef_path, jet_images):
    """
    Run inference on the Hailo-8.
    jet_images: numpy array of shape (N, 3, 32, 32), float32
    Returns: numpy array of shape (N,) with probabilities
    """
    hef = hpf.HEF(hef_path)

    with hpf.VDevice() as target:
        configure_params = hpf.ConfigureParams.create_from_hef(
            hef, interface=hpf.HailoStreamInterface.PCIe
        )
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]

        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )

        print(f"Input:  name={input_vstream_info.name}, shape={input_vstream_info.shape}")
        print(f"Output: name={output_vstream_info.name}, shape={output_vstream_info.shape}")

        results_list = []
        with network_group.activate(network_group_params):
            with hpf.InferVStreams(network_group,
                                    input_vstreams_params,
                                    output_vstreams_params) as infer_pipeline:
                for i in range(len(jet_images)):
                    input_data = {
                        input_vstream_info.name: np.expand_dims(jet_images[i], axis=0)
                    }
                    result = infer_pipeline.infer(input_data)
                    output = result[output_vstream_info.name]
                    results_list.append(output.flatten()[0])

        return np.array(results_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run jet classifier on Hailo-8")
    parser.add_argument("--hef", default="jet_classifier.hef", help="Path to HEF file")
    parser.add_argument("--data", default="QG_jets.npz", help="Path to jet data")
    parser.add_argument("--n", type=int, default=100, help="Number of jets to classify")
    args = parser.parse_args()

    # load jets
    X, y = load_jets(args.data)
    n = min(args.n, len(X))

    # convert to 3-channel images
    print(f"Preprocessing {n} jets...")
    test_images = np.stack([
        jet_to_image_3ch(X[i], R=0.4, npixels=32, pt_min=0.0, normalize=True)
        for i in range(n)
    ])

    # run inference
    print("Running Hailo inference...")
    probs = run_inference(args.hef, test_images)

    # print results
    correct = 0
    for i in range(min(20, n)):
        label = "quark" if y[i] == 1 else "gluon"
        pred  = "quark" if probs[i] > 0.5 else "gluon"
        mark  = "ok" if label == pred else "WRONG"
        print(f"  Jet {i:3d}: true={label:5s}  pred={pred:5s}  prob={probs[i]:.3f}  {mark}")
        if label == pred:
            correct += 1

    # overall accuracy
    preds = (probs > 0.5).astype(int)
    accuracy = (preds == y[:n]).mean()
    print(f"\nOverall accuracy on {n} jets: {accuracy:.4f}")
