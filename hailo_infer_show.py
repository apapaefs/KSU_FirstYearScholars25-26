"""
Interactive Hailo inference viewer for quark/gluon jet classification.

Shows 3D tower plots for each jet (3 channels: positive, negative, neutral charge)
alongside the Hailo classification result. Press Next/Prev buttons or arrow keys
to browse through jets.

Usage:
    PYTHONPATH=/usr/lib/python3/dist-packages python3 hailo_infer_show.py \
        [--hef jet_classifier.hef] [--data QG_jets.npz] [--n 100]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import hailo_platform as hpf
from load_jets import load_jets, jet_to_image_3ch

# Suppress known matplotlib bug: AttributeError in Axes3D._button_release
# when toolbar is None (harmless but noisy on some backends)
import mpl_toolkits.mplot3d.axes3d as _axes3d
_orig_button_release = _axes3d.Axes3D._button_release
def _safe_button_release(self, event):
    try:
        return _orig_button_release(self, event)
    except AttributeError:
        pass
_axes3d.Axes3D._button_release = _safe_button_release


def run_inference(hef_path, jet_images):
    """Run inference on the Hailo-8. jet_images: (N, 32, 32, 3) NHWC float32."""
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


def plot_3ch_towers(fig, img_3ch, R=0.4):
    """
    Plot 3 channels of a jet image as 3D tower plots side by side.
    img_3ch: shape (3, npix, npix) in NCHW format.
    """
    channel_names = [r"Positive ($q=+1$)", r"Negative ($q=-1$)", r"Neutral ($q=0$)"]
    channel_colors = ["#d62728", "#1f77b4", "#2ca02c"]  # red, blue, green

    npix = img_3ch.shape[1]
    x_edges = np.linspace(-R, R, npix + 1)
    y_edges = np.linspace(-R, R, npix + 1)
    x = x_edges[:-1]
    y = y_edges[:-1]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    x0 = xx.ravel()
    y0 = yy.ravel()
    z0 = np.zeros_like(x0)
    dx = (2 * R) / npix
    dy = (2 * R) / npix

    axes = []
    for ch in range(3):
        ax = fig.add_subplot(2, 3, ch + 1, projection="3d")
        dz = img_3ch[ch].ravel()

        # only plot non-zero towers for speed
        mask = dz > 0
        if mask.any():
            ax.bar3d(x0[mask], y0[mask], z0[mask], dx, dy, dz[mask],
                     shade=True, color=channel_colors[ch], alpha=0.8)

        ax.set_xlabel(r"$\Delta y$", fontsize=8)
        ax.set_ylabel(r"$\Delta \phi$", fontsize=8)
        ax.set_zlabel(r"$p_T$ frac", fontsize=8)
        ax.set_title(channel_names[ch], fontsize=10)

        tick_fs = 6
        ax.xaxis.set_tick_params(labelsize=tick_fs)
        ax.yaxis.set_tick_params(labelsize=tick_fs)
        ax.zaxis.set_tick_params(labelsize=tick_fs)
        ax.view_init(elev=25, azim=-60)

        axes.append(ax)

    return axes


def plot_combined_towers(fig, img_3ch, R=0.4):
    """Plot all 3 channels overlaid in a single 3D plot."""
    channel_colors = ["#d62728", "#1f77b4", "#2ca02c"]  # red, blue, green
    channel_labels = ["$q=+1$", "$q=-1$", "$q=0$"]

    npix = img_3ch.shape[1]
    x_edges = np.linspace(-R, R, npix + 1)
    y_edges = np.linspace(-R, R, npix + 1)
    x = x_edges[:-1]
    y = y_edges[:-1]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    x0 = xx.ravel()
    y0 = yy.ravel()
    z0 = np.zeros_like(x0)
    dx = (2 * R) / npix
    dy = (2 * R) / npix

    ax = fig.add_subplot(2, 3, (4, 6), projection="3d")

    for ch in range(3):
        dz = img_3ch[ch].ravel()
        mask = dz > 0
        if mask.any():
            ax.bar3d(x0[mask], y0[mask], z0[mask], dx, dy, dz[mask],
                     shade=True, color=channel_colors[ch], alpha=0.6,
                     label=channel_labels[ch])

    ax.set_xlabel(r"$\Delta y$", fontsize=9)
    ax.set_ylabel(r"$\Delta \phi$", fontsize=9)
    ax.set_zlabel(r"$p_T$ fraction", fontsize=9)
    ax.set_title("All channels combined", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    tick_fs = 7
    ax.xaxis.set_tick_params(labelsize=tick_fs)
    ax.yaxis.set_tick_params(labelsize=tick_fs)
    ax.zaxis.set_tick_params(labelsize=tick_fs)
    ax.view_init(elev=25, azim=-60)

    return ax


class JetViewer:
    def __init__(self, images_nchw, labels, probs):
        self.images = images_nchw  # (N, 3, 32, 32)
        self.labels = labels
        self.probs = probs
        self.n = len(images_nchw)
        self.idx = 0

        self.fig = plt.figure(figsize=(14, 9))
        self.fig.canvas.manager.set_window_title("Quark/Gluon Jet Classifier - Hailo Inference")

        # buttons
        ax_prev = self.fig.add_axes([0.25, 0.01, 0.12, 0.04])
        ax_next = self.fig.add_axes([0.63, 0.01, 0.12, 0.04])
        self.btn_prev = Button(ax_prev, "< Prev")
        self.btn_next = Button(ax_next, "Next >")
        self.btn_prev.on_clicked(self.prev_jet)
        self.btn_next.on_clicked(self.next_jet)

        # keyboard navigation
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.draw_jet()
        plt.show()

    def draw_jet(self):
        # clear all axes except button axes
        for ax in self.fig.get_axes():
            if ax not in [self.btn_prev.ax, self.btn_next.ax]:
                ax.remove()

        # classification info
        true_label = "Quark" if self.labels[self.idx] == 1 else "Gluon"
        pred_label = "Quark" if self.probs[self.idx] > 0.5 else "Gluon"
        prob = self.probs[self.idx]
        correct = true_label == pred_label
        status = "CORRECT" if correct else "WRONG"
        status_color = "green" if correct else "red"

        self.fig.suptitle(
            f"Jet {self.idx}/{self.n-1}    |    "
            f"True: {true_label}    |    "
            f"Predicted: {pred_label} (prob={prob:.3f})    |    "
            f"{status}",
            fontsize=13, fontweight="bold", color=status_color,
            y=0.98
        )

        # plot the 3 channels + combined
        img = self.images[self.idx]  # (3, 32, 32)
        plot_3ch_towers(self.fig, img, R=0.4)
        plot_combined_towers(self.fig, img, R=0.4)

        self.fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.3, wspace=0.3)
        self.fig.canvas.draw_idle()

    def next_jet(self, event=None):
        self.idx = (self.idx + 1) % self.n
        self.draw_jet()

    def prev_jet(self, event=None):
        self.idx = (self.idx - 1) % self.n
        self.draw_jet()

    def on_key(self, event):
        if event.key in ("right", "n"):
            self.next_jet()
        elif event.key in ("left", "p"):
            self.prev_jet()
        elif event.key == "q":
            plt.close(self.fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Hailo jet classifier viewer")
    parser.add_argument("--hef", default="jet_classifier.hef", help="Path to HEF file")
    parser.add_argument("--data", default="QG_jets.npz", help="Path to jet data")
    parser.add_argument("--n", type=int, default=100, help="Number of jets to classify")
    args = parser.parse_args()

    # load jets
    X, y = load_jets(args.data)
    n = min(args.n, len(X))

    # convert to 3-channel images (keep NCHW for plotting)
    print(f"Preprocessing {n} jets...")
    images_nchw = np.stack([
        jet_to_image_3ch(X[i], R=0.4, npixels=32, pt_min=0.0, normalize=True)
        for i in range(n)
    ])

    # convert to NHWC for Hailo inference
    images_nhwc = np.transpose(images_nchw, (0, 2, 3, 1))

    # run Hailo inference
    print("Running Hailo inference...")
    probs = run_inference(args.hef, images_nhwc)
    print(f"Inference complete on {n} jets.")

    # overall accuracy
    preds = (probs > 0.5).astype(int)
    accuracy = (preds == y[:n]).mean()
    print(f"Overall accuracy: {accuracy:.4f}")

    # launch interactive viewer
    print("Launching viewer... (arrow keys or Next/Prev buttons to navigate, Q to quit)")
    viewer = JetViewer(images_nchw, y[:n], probs)
