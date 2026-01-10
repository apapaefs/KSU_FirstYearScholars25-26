import numpy as np
import pickle

path = "QG_jets.npz"  # Pythia Jets
# path -= QG_jets_herwig_0.npz # Herwig Jets

# Load 
data = np.load(path, allow_pickle=True)  # allow_pickle only if object arrays are stored

# print here:
# X: (100000,M,4), exactly 50k quark and 50k gluon jets, randomly sorted, where M is the max multiplicity of the jets in that file (other jets have been padded with zero-particles), and the features of each particle are its pt, rapidity, azimuthal angle, and pdgid.
# y: (100000,), an array of labels for the jets where gluon is 0 and quark is 1.
print("Keys in file:", data.files)
for k in data.files:
    arr = data[k]
    print(f"{k:>20s}  shape={arr.shape}  dtype={arr.dtype}")
