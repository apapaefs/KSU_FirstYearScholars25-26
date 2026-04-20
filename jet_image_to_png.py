#!/usr/bin/env python3
"""
Render jet images from preprocessed .npz files as PNG figures.

Reads a jet_images_3ch_*.npz file (keys: 'images' shape (N,3,32,32), 'y')
and saves PNG files showing the 3-channel jet image as a 2D grid.

Usage:
    # Save a few example quark and gluon jets:
    python3 jet_image_to_png.py output/jet_images_3ch_8-16-32.npz \
        --n 5 --outdir plots/jet_images

    # Save a specific jet by index:
    python3 jet_image_to_png.py output/jet_images_3ch_8-16-32.npz \
        --indices 0 42 99 --outdir plots/jet_images

    # Show the average jet image for quarks and gluons:
    python3 jet_image_to_png.py output/jet_images_3ch_8-16-32.npz \
        --average --outdir plots/jet_images
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_jet_image(img, label, title, outpath, R=0.4):
    """
    Plot a single jet image (3, 32, 32) as a figure with 4 panels:
    combined RGB, and each charge channel separately.

    Parameters
    ----------
    img   : ndarray (3, 32, 32) — [positive, negative, neutral] pT channels
    label : int (0=gluon, 1=quark) or None
    title : str
    outpath : str — output PNG path
    R     : float — jet radius (sets axis range)
    """
    npix = img.shape[1]
    extent = [-R, R, -R, R]

    ch_pos = img[0]   # positive charge
    ch_neg = img[1]   # negative charge
    ch_neu = img[2]   # neutral

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    label_str = {0: "gluon", 1: "quark"}.get(label, "unknown")

    # Use reversed colormaps so zero = white
    ch_labels = [r"$p_T$ (charge $+$)", r"$p_T$ (charge $-$)", r"$p_T$ (neutral)"]
    cmaps = ["Reds", "Blues", "Greens"]
    for i, (ch, cmap, ch_label) in enumerate(zip(
            [ch_pos, ch_neg, ch_neu], cmaps, ch_labels)):
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="white")
        masked = np.ma.masked_equal(ch, 0.0)
        im = axes[i].imshow(masked, origin="lower", extent=extent,
                            cmap=cmap_obj, interpolation="nearest",
                            vmin=0)
        axes[i].set_title(ch_label)
        axes[i].set_xlabel(r"$\Delta \eta$")
        if i == 0:
            axes[i].set_ylabel(r"$\Delta \phi$")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle(f"{title}  [{label_str}]", fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_average_image(images, labels, outdir, R=0.4):
    """Plot average jet image for quarks and gluons side by side."""
    for label, name in [(1, "quark"), (0, "gluon")]:
        mask = labels == label
        if mask.sum() == 0:
            print(f"  No {name} jets found, skipping average")
            continue
        avg = images[mask].mean(axis=0)
        plot_jet_image(avg, label,
                       f"Average {name} jet ({mask.sum()} jets)",
                       os.path.join(outdir, f"avg_{name}.png"), R=R)


def main():
    parser = argparse.ArgumentParser(
        description="Render jet images from preprocessed .npz as PNGs")
    parser.add_argument("input", type=str,
                        help="Path to jet_images_3ch_*.npz file")
    parser.add_argument("--outdir", type=str, default="plots/jet_images",
                        help="Output directory for PNGs (default: plots/jet_images)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of example jets per class to plot (default: 5)")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Specific jet indices to plot (overrides --n)")
    parser.add_argument("--average", action="store_true",
                        help="Plot average jet images for each class")
    parser.add_argument("--R", type=float, default=0.4,
                        help="Jet radius for axis labels (default: 0.4)")

    args = parser.parse_args()

    data = np.load(args.input)
    images = data["images"]  # (N, 3, 32, 32)
    labels = data["y"]       # (N,)

    print(f"Loaded {len(images)} jets from {args.input}")
    print(f"  Shape: {images.shape}, quarks: {(labels==1).sum()}, "
          f"gluons: {(labels==0).sum()}")

    if args.average:
        plot_average_image(images, labels, args.outdir, R=args.R)

    if args.indices is not None:
        for idx in args.indices:
            if idx >= len(images):
                print(f"  Index {idx} out of range, skipping")
                continue
            plot_jet_image(images[idx], int(labels[idx]),
                           f"Jet #{idx}",
                           os.path.join(args.outdir, f"jet_{idx}.png"),
                           R=args.R)
    else:
        # Plot n examples of each class
        for label, name in [(1, "quark"), (0, "gluon")]:
            idxs = np.where(labels == label)[0]
            if len(idxs) == 0:
                continue
            # Pick evenly spaced examples
            chosen = idxs[np.linspace(0, len(idxs) - 1, args.n, dtype=int)]
            for i, idx in enumerate(chosen):
                plot_jet_image(images[idx], label,
                               f"{name.capitalize()} jet #{idx}",
                               os.path.join(args.outdir,
                                            f"{name}_{i:02d}_idx{idx}.png"),
                               R=args.R)


if __name__ == "__main__":
    main()
