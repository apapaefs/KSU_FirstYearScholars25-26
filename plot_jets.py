import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

from load_jets import load_jets, jet_to_image

#################
# PLOT FUNCTION #
#################

def plot_jet_image_towers(img, R=0.4, zlabel=r'$\sum p_T$'):
    """
    img: (npix, npix) array
    R: half-width of the (Δy, Δφ) window used to make the image
    """
    npix = img.shape[0]
    assert img.shape == (npix, npix)

    # Coordinates of the lower-left corner of each "tower"
    x_edges = np.linspace(-R, R, npix + 1)   # Δy edges
    y_edges = np.linspace(-R, R, npix + 1)   # Δφ edges

    # get the edges
    x = x_edges[:-1]
    y = y_edges[:-1]
    # create a grid
    xx, yy = np.meshgrid(x, y, indexing="ij")

    x0 = xx.ravel()
    y0 = yy.ravel()
    z0 = np.zeros_like(x0)

    dx = (2 * R) / npix
    dy = (2 * R) / npix
    dz = img.ravel()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.bar3d(x0, y0, z0, dx, dy, dz, shade=True)

    ax.set_xlabel(r'$\Delta y$',fontsize=10)
    ax.set_ylabel(r'$\Delta \phi$',fontsize=10)
    ax.set_zlabel(zlabel)
    ax.set_title("Jet image as 3D towers")


    # change font size for tick labels
    tick_fs = 7
    ax.xaxis.set_tick_params(labelsize=tick_fs)
    ax.yaxis.set_tick_params(labelsize=tick_fs)
    ax.zaxis.set_tick_params(labelsize=tick_fs)

    # Optional: nicer view angle
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0)          # make pane transparent
        axis.pane.set_edgecolor((0,0,0,0))  # optional: hide pane border

    # remove grid lines

    return fig, ax

########################
# PLOTTING STARTS HERE #
########################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a jet as 3D towers")
    parser.add_argument("--data", type=str, default="data/QG_jets.npz",
                        help="Path to jet data file")
    parser.add_argument("--outdir", type=str, default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--jet", type=int, default=5, help="Jet index to plot")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X, y = load_jets(args.data)
    img = jet_to_image(X[args.jet], npixels=32)
    fig, ax = plot_jet_image_towers(img, R=0.4)

    out_path = os.path.join(args.outdir, "jet_towers.pdf")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.30, transparent=True)
    print(f"Saved {out_path}")
    plt.close(fig)
