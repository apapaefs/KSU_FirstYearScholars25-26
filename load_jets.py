import numpy as np
import pickle
import sys

#####################
# FUNCTIONS GO HERE # 
#####################

# define function to load the data: 
def load_jets(path):
    """
    Loads the jet data from an .npz file
    """
    # Load the data
    data = np.load(path, allow_pickle=True)  # allow_pickle only if object arrays are stored
    # print here:
    # X: (100000,M,4), exactly 50k quark and 50k gluon jets, randomly sorted, where M is the max multiplicity of the jets in that file (other jets have been padded with zero-particles), and the features of each particle are its pt, rapidity, azimuthal angle, and pdgid.
    # y: (100000,), an array of labels for the jets where gluon is 0 and quark is 1.
    print("Keys in file:", data.files)
    for k in data.files:
        arr = data[k]
        print(f"{k:>20s}  shape={arr.shape}  dtype={arr.dtype}")
    X = data["X"]      # e.g. (N, M, 4)
    y = data["y"]      # e.g. (N,)
    return X, y

# Delta phi has to lie in [-pi, pi)
def delta_phi(phi, phi0):
    d = phi - phi0
    return np.arctan2(np.sin(d), np.cos(d))  # wraps to [-pi, pi)

# get the jet axis:
def jet_axis_from_constituents(jet, pt_min=0.0):
    """
    jet: array of shape (n_const, 4) with columns [pt, y, phi, pdgid]
    returns: (y_jet, phi_jet)
    """

    # get the pt, y and phi of all particles in a jet
    pt  = jet[:, 0]
    y   = jet[:, 1]
    phi = jet[:, 2]

    # remove empty entries or entries with low pT
    m = pt > pt_min
    pt, y, phi = pt[m], y[m], phi[m]

    # if there's no particles left return zeros
    if pt.size == 0:
        return 0.0, 0.0  # degenerate case

    # convert the pt, y and phi into a four-vector
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(y)
    E  = pt * np.cosh(y)

    # sum up all components of all particles
    Px = px.sum()
    Py = py.sum()
    Pz = pz.sum()
    E  = E.sum()

    # get the phi of the jet 
    phi_jet = np.arctan2(Py, Px)

    # get the rapidity of the jet. use a cutoff for numerical safety
    eps = 1e-12
    y_jet = 0.5 * np.log((E + Pz + eps) / (E - Pz + eps))

    return y_jet, phi_jet

# takes a jet and constructs the jet image 
def jet_to_image(jet, R=0.4, npixels=32, pt_min=0.0, normalize=True):
    """
    Returns image of shape (npixels, npixels) where each pixel is sum(pt) in that bin.
    """
    
    # get the pt, y and phi of all particles in a jet
    pt  = jet[:, 0]
    y   = jet[:, 1]
    phi = jet[:, 2]

    # remove empty entries or entries with low pT
    m = pt > pt_min
    pt, y, phi = pt[m], y[m], phi[m]

    # if there's no particles left return empty image
    if pt.size == 0:
        return np.zeros((npix, npix), dtype=np.float32)

    # get the jet axis information 
    y0, phi0 = jet_axis_from_constituents(jet, pt_min=pt_min)
    
    # subtract the jet axis (y, phi) from the constituent (y, phi)
    dy  = y - y0
    dph = delta_phi(phi, phi0)

    # Keep constituents inside square window defined by R x R
    keep = (np.abs(dy) <= R) & (np.abs(dph) <= R)
    dy, dph, pt = dy[keep], dph[keep], pt[keep]

    # Convert dy,dphi -> integer pixel indices
    # Map [-R, R] to [0, npixels)
    ix = np.floor((dy  + R) * (npixels / (2 * R))).astype(int)
    iy = np.floor((dph + R) * (npixels / (2 * R))).astype(int)
    valid = (ix >= 0) & (ix < npixels) & (iy >= 0) & (iy < npixels)
    ix, iy, pt = ix[valid], iy[valid], pt[valid]

    # start with empty image
    img = np.zeros((npixels, npixels), dtype=np.float32)

    # Accumulate pt into pixels
    np.add.at(img, (ix, iy), pt.astype(np.float32))

    # if we want to normalize then divide by the sum of all pixels
    if normalize:
        s = img.sum()
        if s > 0:
            img /= s

    return img


def plot_jet_image_towers(img, R=0.4, zlabel=r'$\sum p_T$'):
    """
    img: (npix, npix) array, e.g. output of jet_to_image_32x32(..., normalize=False)
    R: half-width of the (Δy, Δφ) window used to make the image
    """
    npix = img.shape[0]
    assert img.shape == (npix, npix)

    # Coordinates of the lower-left corner of each "tower"
    x_edges = np.linspace(-R, R, npix + 1)   # Δy edges
    y_edges = np.linspace(-R, R, npix + 1)   # Δφ edges

    x = x_edges[:-1]
    y = y_edges[:-1]
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

    ax.set_xlabel(r'$\Delta y$')
    ax.set_ylabel(r'$\Delta \phi$')
    ax.set_zlabel(zlabel)
    ax.set_title("Jet image as 3D towers")

    # Optional: nicer view angle
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.show()

####################################
# Load some jet data and do things #
####################################

jetpath = "QG_jets.npz"  # Pythia Jets
# path -= QG_jets_herwig_0.npz # Herwig Jets
X, y = load_jets(jetpath)

# test: convert jet to image:
#print(jet_to_image(X[0],npixels=6))


