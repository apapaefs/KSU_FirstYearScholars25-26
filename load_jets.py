import numpy as np

#####################
# FUNCTIONS GO HERE #
#####################

# PDG ID -> electric charge mapping for common jet constituents
PDG_TO_CHARGE = {
    # Charged pions
    211: +1, -211: -1,
    # Neutral pion
    111: 0,
    # Charged kaons
    321: +1, -321: -1,
    # Neutral kaons
    311: 0, 130: 0, 310: 0,
    # Protons / antiprotons
    2212: +1, -2212: -1,
    # Neutrons
    2112: 0, -2112: 0,
    # Photon
    22: 0,
    # Electrons
    11: -1, -11: +1,
    # Muons
    13: -1, -13: +1,
    # Taus
    15: -1, -15: +1,
    # Neutrinos
    12: 0, -12: 0, 14: 0, -14: 0, 16: 0, -16: 0,
    # Sigma baryons
    3222: +1, -3222: -1,
    3112: -1, -3112: +1,
    3212: 0, -3212: 0,
    # Lambda
    3122: 0, -3122: 0,
    # Xi baryons
    3312: -1, -3312: +1,
    3322: 0, -3322: 0,
    # Omega
    3334: -1, -3334: +1,
    # D mesons
    411: +1, -411: -1,
    421: 0, -421: 0,
    # B mesons
    521: +1, -521: -1,
    511: 0, -511: 0,
}

# vectorized charge lookup (unknown PDG IDs default to neutral)
_pdgid_to_charge_vec = np.vectorize(lambda pid: PDG_TO_CHARGE.get(int(pid), 0))

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


# takes a jet and constructs a 3-channel jet image using charge information
def jet_to_image_3ch(jet, R=0.4, npixels=32, pt_min=0.0, normalize=True):
    """
    Returns image of shape (3, npixels, npixels).
    Channel 0: sum pT of positively charged particles
    Channel 1: sum pT of negatively charged particles
    Channel 2: sum pT of neutral particles
    """

    # get the pt, y, phi and pdgid of all particles in a jet
    pt    = jet[:, 0]
    y     = jet[:, 1]
    phi   = jet[:, 2]
    pdgid = jet[:, 3]

    # remove empty entries or entries with low pT
    m = pt > pt_min
    pt, y, phi, pdgid = pt[m], y[m], phi[m], pdgid[m]

    # if there's no particles left return empty image
    if pt.size == 0:
        return np.zeros((3, npixels, npixels), dtype=np.float32)

    # get the jet axis information
    y0, phi0 = jet_axis_from_constituents(jet, pt_min=pt_min)

    # subtract the jet axis (y, phi) from the constituent (y, phi)
    dy  = y - y0
    dph = delta_phi(phi, phi0)

    # Keep constituents inside square window defined by R x R
    keep = (np.abs(dy) <= R) & (np.abs(dph) <= R)
    dy, dph, pt, pdgid = dy[keep], dph[keep], pt[keep], pdgid[keep]

    # Convert dy,dphi -> integer pixel indices
    # Map [-R, R] to [0, npixels)
    ix = np.floor((dy  + R) * (npixels / (2 * R))).astype(int)
    iy = np.floor((dph + R) * (npixels / (2 * R))).astype(int)
    valid = (ix >= 0) & (ix < npixels) & (iy >= 0) & (iy < npixels)
    ix, iy, pt, pdgid = ix[valid], iy[valid], pt[valid], pdgid[valid]

    # Map PDG IDs to charges (+1, -1, or 0)
    charges = _pdgid_to_charge_vec(pdgid)

    # start with empty 3-channel image
    img = np.zeros((3, npixels, npixels), dtype=np.float32)

    # Channel 0: positive charge
    mask_pos = charges > 0
    if mask_pos.any():
        np.add.at(img[0], (ix[mask_pos], iy[mask_pos]), pt[mask_pos].astype(np.float32))

    # Channel 1: negative charge
    mask_neg = charges < 0
    if mask_neg.any():
        np.add.at(img[1], (ix[mask_neg], iy[mask_neg]), pt[mask_neg].astype(np.float32))

    # Channel 2: neutral
    mask_neu = charges == 0
    if mask_neu.any():
        np.add.at(img[2], (ix[mask_neu], iy[mask_neu]), pt[mask_neu].astype(np.float32))

    # if we want to normalize then divide by the sum of all pixels across all channels
    if normalize:
        s = img.sum()
        if s > 0:
            img /= s

    return img


def preprocess_all_jets(X, R=0.4, npixels=32, pt_min=0.0, normalize=True):
    """Convert all jets to 3-channel images. Returns array of shape (N, 3, npixels, npixels)."""
    N = X.shape[0]
    images = np.zeros((N, 3, npixels, npixels), dtype=np.float32)
    for i in range(N):
        images[i] = jet_to_image_3ch(X[i], R=R, npixels=npixels,
                                      pt_min=pt_min, normalize=normalize)
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{N} jets")
    return images



if __name__ == "__main__":
    # Quick test: load jets and print info
    jetpath = "data/QG_jets.npz"
    X, y = load_jets(jetpath)
    print(f"Loaded {len(X)} jets")
    print(f"  First jet shape: {X[0].shape}")
    img = jet_to_image_3ch(X[0], npixels=32)
    print(f"  3ch image shape: {img.shape}")
