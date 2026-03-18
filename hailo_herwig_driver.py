"""
Concurrent Herwig + Hailo Z+jet classification driver.

Runs Herwig 7 in batches (each with a different seed), producing ROOT files.
As each ROOT file completes, reads it, finds jets with anti-kT R=0.4,
matches the leading jet to parton-level truth, builds a 3-channel jet image,
and classifies it as quark or gluon using the Hailo-8.

The Z boson decays to neutrinos (invisible), so each event has one hard jet.

Usage (on Raspberry Pi 5 with Hailo-8):
    PYTHONPATH=/usr/lib/python3/dist-packages python3 hailo_herwig_driver.py \
        --tag 3ch_16-32-64 \
        --run-file LHC-Zjet.run \
        --workdir Herwig/ \
        --n-batches 10 --batch-size 1000 --seed-start 1

Requirements:
    pip install uproot fastjet awkward numpy
"""

# ──────────────────────────────────────────────────────────────────────
# Fix LD_LIBRARY_PATH conflict: pip fastjet (3.4.3) vs Herwig's bundled
# libfastjet (3.4.0).  `module load herwig/stable` puts Herwig's lib dir
# on LD_LIBRARY_PATH, which shadows the pip-installed fastjet's own
# bundled library and causes an undefined-symbol error at import time.
#
# Strategy: on first invocation, save the full LD_LIBRARY_PATH for the
# Herwig subprocess, strip Herwig paths, and re-exec so the dynamic
# linker picks up the correct libfastjet for Python.
# ──────────────────────────────────────────────────────────────────────
import os
import sys

_HERWIG_ENV_KEY = '_HERWIG_ORIG_LD_LIBRARY_PATH'
if _HERWIG_ENV_KEY not in os.environ:
    os.environ[_HERWIG_ENV_KEY] = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['_HERWIG_ORIG_PATH'] = os.environ.get('PATH', '')
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    filtered = ':'.join(p for p in ld_path.split(':')
                        if p and 'herwig' not in p.lower())
    if filtered:
        os.environ['LD_LIBRARY_PATH'] = filtered
    elif 'LD_LIBRARY_PATH' in os.environ:
        del os.environ['LD_LIBRARY_PATH']
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import glob
import threading
import queue
import subprocess
import time
import numpy as np

import uproot
import awkward as ak
import fastjet

from load_jets import jet_to_image_3ch
from hailo_infer import run_inference


# ──────────────────────────────────────────────────────────────────────
# Kinematics helpers
# ──────────────────────────────────────────────────────────────────────

def four_vector_to_ptyphiid(E, px, py, pz, pdgid):
    """
    Convert (E, px, py, pz, pdgid) arrays to (pt, y, phi, pdgid).

    Parameters
    ----------
    E, px, py, pz : 1-D numpy arrays of length N
    pdgid         : 1-D numpy array of length N (integer-valued)

    Returns
    -------
    result : ndarray of shape (N, 4) with columns [pt, y, phi, pdgid]
    """
    pt = np.sqrt(px**2 + py**2)
    eps = 1e-12
    y = 0.5 * np.log((E + pz + eps) / (E - pz + eps))
    phi = np.arctan2(py, px)
    return np.column_stack([pt, y, phi, pdgid])


def delta_R(y1, phi1, y2, phi2):
    """Compute DeltaR between two objects given rapidity and phi."""
    dphi = phi1 - phi2
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))  # wrap to [-pi, pi)
    return np.sqrt((y1 - y2)**2 + dphi**2)


# ──────────────────────────────────────────────────────────────────────
# PDG IDs to exclude from jet finding: neutrinos + EW bosons
# (Herwig may store intermediate particles in the objects array)
# ──────────────────────────────────────────────────────────────────────
NEUTRINO_IDS = {12, -12, 14, -14, 16, -16}
EW_BOSON_IDS = {22, 23, 24, -24, 25}  # gamma, Z, W+, W-, H
EXCLUDE_FROM_JETS = NEUTRINO_IDS | EW_BOSON_IDS


# ──────────────────────────────────────────────────────────────────────
# ROOT file reader
# ──────────────────────────────────────────────────────────────────────

def read_herwig_root(filepath):
    """
    Read a Herwig HwSim ROOT file and return a list of event dicts.

    Each dict contains:
        particles : ndarray (n_part, 5) — [E, px, py, pz, pdgid]
        partons   : ndarray (n_out, 5)  — [E, px, py, pz, pdgid]
        incoming  : ndarray (2, 5)      — [E, px, py, pz, pdgid]
        weight    : float
    """
    f = uproot.open(filepath)
    tree = f["Data"]

    # Read all branches we need
    numparticles = tree["numparticles"].array(library="np")
    objects_raw = tree["objects"].array(library="np")
    evweight = tree["evweight"].array(library="np")
    numoutgoing = tree["numoutgoing"].array(library="np")
    partons_raw = tree["partons"].array(library="np")
    incoming_raw = tree["incoming"].array(library="np")

    n_events = len(numparticles)
    events = []

    for i in range(n_events):
        npart = int(numparticles[i])
        nout = int(numoutgoing[i])

        # objects_raw[i] has shape (8, 10000) — rows: E, px, py, pz, pdgid, charge, ...
        # Extract only the first npart columns
        obj = objects_raw[i]  # shape (8, 10000)
        particles = np.zeros((npart, 5), dtype=np.float64)
        particles[:, 0] = obj[0, :npart]  # E
        particles[:, 1] = obj[1, :npart]  # px
        particles[:, 2] = obj[2, :npart]  # py
        particles[:, 3] = obj[3, :npart]  # pz
        particles[:, 4] = obj[4, :npart]  # pdgid

        # partons_raw[i] has shape (5, 100) — rows: E, px, py, pz, pdgid
        par = partons_raw[i]  # shape (5, 100)
        partons = np.zeros((nout, 5), dtype=np.float64)
        partons[:, 0] = par[0, :nout]  # E
        partons[:, 1] = par[1, :nout]  # px
        partons[:, 2] = par[2, :nout]  # py
        partons[:, 3] = par[3, :nout]  # pz
        partons[:, 4] = par[4, :nout]  # pdgid

        # incoming_raw[i] has shape (5, 2) — rows: E, px, py, pz, pdgid
        inc = incoming_raw[i]  # shape (5, 2)
        incoming = np.zeros((2, 5), dtype=np.float64)
        incoming[:, 0] = inc[0, :2]
        incoming[:, 1] = inc[1, :2]
        incoming[:, 2] = inc[2, :2]
        incoming[:, 3] = inc[3, :2]
        incoming[:, 4] = inc[4, :2]

        events.append({
            "particles": particles,
            "partons": partons,
            "incoming": incoming,
            "weight": float(evweight[i]),
        })

    return events


# ──────────────────────────────────────────────────────────────────────
# Jet finding
# ──────────────────────────────────────────────────────────────────────

def find_jets(particles, R=0.4, pt_min_jet=20.0, eta_max=5.0, pt_min_part=0.1):
    """
    Run anti-kT jet clustering on an array of particles.

    Parameters
    ----------
    particles : ndarray (N, 5) with columns [E, px, py, pz, pdgid]
    R         : jet radius
    pt_min_jet: minimum jet pT in GeV
    eta_max   : maximum particle |eta| for input to jet finder
    pt_min_part: minimum particle pT in GeV

    Returns
    -------
    jets_out : list of dicts, sorted by pT (descending), each containing:
        pt, y, phi, mass : jet kinematics
        constituents     : ndarray (n_const, 4) with columns [pt, y, phi, pdgid]
                           (format expected by jet_to_image_3ch)
    """
    E = particles[:, 0]
    px = particles[:, 1]
    py = particles[:, 2]
    pz = particles[:, 3]
    pdgid = particles[:, 4].astype(int)

    # Compute pt and eta for filtering
    pt = np.sqrt(px**2 + py**2)
    theta = np.arctan2(pt, pz)
    eta = -np.log(np.tan(theta / 2.0 + 1e-20))

    # Filter: remove neutrinos + EW bosons, apply pt and eta cuts
    mask = np.ones(len(particles), dtype=bool)
    for pid in EXCLUDE_FROM_JETS:
        mask &= (pdgid != pid)
    mask &= (pt > pt_min_part)
    mask &= (np.abs(eta) < eta_max)

    if mask.sum() == 0:
        return []

    filt_E = E[mask]
    filt_px = px[mask]
    filt_py = py[mask]
    filt_pz = pz[mask]
    filt_pdgid = pdgid[mask]

    # Build awkward array for scikit-hep fastjet
    pj_array = ak.zip({
        "px": filt_px,
        "py": filt_py,
        "pz": filt_pz,
        "E": filt_E,
    })

    # Cluster with anti-kT
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    cluster_seq = fastjet.ClusterSequence(pj_array, jetdef)

    # Get jets and their constituent indices (indices into our filtered array)
    jets = cluster_seq.inclusive_jets(min_pt=pt_min_jet)
    const_indices = cluster_seq.constituent_index(min_pt=pt_min_jet)

    # Convert jets to numpy for sorting
    jet_px = ak.to_numpy(jets["px"])
    jet_py = ak.to_numpy(jets["py"])
    jet_pz = ak.to_numpy(jets["pz"])
    jet_E = ak.to_numpy(jets["E"])
    jet_pt = np.sqrt(jet_px**2 + jet_py**2)

    # Sort by pT descending
    pt_order = np.argsort(-jet_pt)

    # Pre-compute filtered particle kinematics for constituent lookup
    filt_ptyphiid = four_vector_to_ptyphiid(filt_E, filt_px, filt_py, filt_pz, filt_pdgid)

    jets_out = []
    for idx in pt_order:
        jpt = jet_pt[idx]
        jpx = jet_px[idx]
        jpy = jet_py[idx]
        jpz = jet_pz[idx]
        jE = jet_E[idx]

        # Jet rapidity and phi
        eps = 1e-12
        jy = 0.5 * np.log((jE + jpz + eps) / (jE - jpz + eps))
        jphi = np.arctan2(jpy, jpx)
        jmass2 = jE**2 - jpx**2 - jpy**2 - jpz**2
        jmass = np.sqrt(max(jmass2, 0.0))

        # Get constituent indices for this jet
        cidx = ak.to_numpy(const_indices[idx])
        if len(cidx) == 0:
            continue

        # Build constituent array in [pt, y, phi, pdgid] format
        const_arr = filt_ptyphiid[cidx]  # shape (n_const, 4)

        jets_out.append({
            "pt": float(jpt),
            "y": float(jy),
            "phi": float(jphi),
            "mass": float(jmass),
            "constituents": const_arr,
        })

    return jets_out


# ──────────────────────────────────────────────────────────────────────
# Parton truth matching
# ──────────────────────────────────────────────────────────────────────

def match_jets_to_partons(jets, partons, dR_max=0.4):
    """
    Match reconstructed jets to outgoing hard-process partons by DeltaR.

    Parameters
    ----------
    jets    : list of jet dicts (each has 'y', 'phi')
    partons : ndarray (n_out, 5) with columns [E, px, py, pz, pdgid]
    dR_max  : maximum DeltaR for matching

    Returns
    -------
    labels : list of int, one per jet
             1 = quark, 0 = gluon, -1 = unmatched/unknown
    parton_pdgids : list of int, the matched parton's PDG ID (0 if unmatched)
    """
    if len(partons) == 0:
        return [-1] * len(jets), [0] * len(jets)

    # Convert partons to pt, y, phi
    par_ptyphiid = four_vector_to_ptyphiid(
        partons[:, 0], partons[:, 1], partons[:, 2], partons[:, 3], partons[:, 4]
    )
    par_y = par_ptyphiid[:, 1]
    par_phi = par_ptyphiid[:, 2]
    par_pdgid = partons[:, 4].astype(int)

    labels = []
    matched_pdgids = []
    used_partons = set()

    for jet in jets:
        best_dR = 999.0
        best_idx = -1

        for ip in range(len(partons)):
            if ip in used_partons:
                continue
            dR = delta_R(jet["y"], jet["phi"], par_y[ip], par_phi[ip])
            if dR < best_dR:
                best_dR = dR
                best_idx = ip

        if best_dR < dR_max and best_idx >= 0:
            used_partons.add(best_idx)
            pid = abs(par_pdgid[best_idx])
            matched_pdgids.append(int(par_pdgid[best_idx]))
            if 1 <= pid <= 5:
                labels.append(1)   # quark
            elif pid == 21:
                labels.append(0)   # gluon
            else:
                labels.append(-1)  # unknown (e.g. Z with pdgid=23)
        else:
            labels.append(-1)
            matched_pdgids.append(0)

    return labels, matched_pdgids


# ──────────────────────────────────────────────────────────────────────
# Jet classification via Hailo
# ──────────────────────────────────────────────────────────────────────

def classify_jets(jet_constituents_list, hef_path, R=0.4, return_images=False):
    """
    Build 3-channel jet images and run Hailo inference.

    Parameters
    ----------
    jet_constituents_list : list of ndarray, each (n_const, 4) [pt, y, phi, pdgid]
    hef_path              : path to Hailo HEF file
    R                     : jet radius for image construction
    return_images         : if True, also return the NCHW images for display

    Returns
    -------
    probs : ndarray of shape (n_jets,) with quark probabilities
    images_nchw : (only if return_images) list of ndarray (3, 32, 32)
    """
    if len(jet_constituents_list) == 0:
        return (np.array([]), []) if return_images else np.array([])

    # Build images: NCHW (3, 32, 32) then transpose to NHWC (32, 32, 3) for Hailo
    images_nchw = []
    images_nhwc = []
    for const in jet_constituents_list:
        img = jet_to_image_3ch(const, R=R, npixels=32, pt_min=0.0, normalize=True)
        images_nchw.append(img)
        img_nhwc = np.transpose(img, (1, 2, 0))  # (3,32,32) -> (32,32,3)
        images_nhwc.append(img_nhwc)

    images_nhwc = np.stack(images_nhwc).astype(np.float32)

    # Run Hailo inference
    probs = run_inference(hef_path, images_nhwc)
    if return_images:
        return probs, images_nchw
    return probs


# ──────────────────────────────────────────────────────────────────────
# Process a single ROOT file
# ──────────────────────────────────────────────────────────────────────

def process_root_file(filepath, hef_path, jet_R=0.4, jet_pt_min=500.0,
                      jet_pt_max=550.0, jet_y_max=1.7, show=False):
    """
    Read a Herwig ROOT file, find the leading jet, match truth, classify.

    Parameters
    ----------
    show : bool
        If True, also return NCHW jet images for interactive display.

    Returns
    -------
    results : list of dicts, one per event with >=1 jet passing cuts:
        jet_pt, jet_y, jet_phi, jet_prob, jet_truth, jet_pdgid, weight
        If show=True, also includes 'jet_image' (ndarray shape (3,32,32))
    n_total : total number of events in the file
    """
    events = read_herwig_root(filepath)
    n_total = len(events)
    results = []

    for evt in events:
        # Find jets (use a low clustering threshold; fiducial cuts applied below)
        jets = find_jets(evt["particles"], R=jet_R, pt_min_jet=20.0)

        # Need at least 1 jet
        if len(jets) < 1:
            continue

        # Take the leading jet
        j = jets[0]

        # Apply fiducial cuts
        if j["pt"] < jet_pt_min or j["pt"] > jet_pt_max:
            continue
        if abs(j["y"]) > jet_y_max:
            continue

        # Match to parton truth (exclude EW bosons: Z=23, W=24, H=25, gamma=22)
        ew_ids = {22, 23, 24, 25}
        qcd_partons = np.array([p for p in evt["partons"]
                                if abs(int(p[4])) not in ew_ids])
        labels, pdgids = match_jets_to_partons(
            [j], qcd_partons if len(qcd_partons) > 0 else evt["partons"][:0],
            dR_max=jet_R
        )

        # Classify the jet (optionally return images for display)
        if show:
            probs, images_nchw = classify_jets(
                [j["constituents"]], hef_path, R=jet_R, return_images=True
            )
        else:
            probs = classify_jets([j["constituents"]], hef_path, R=jet_R)

        result = {
            "jet_pt": j["pt"], "jet_y": j["y"], "jet_phi": j["phi"],
            "jet_prob": float(probs[0]), "jet_truth": labels[0],
            "jet_pdgid": pdgids[0],
            "weight": evt["weight"],
        }
        if show:
            result["jet_image"] = images_nchw[0]  # (3, 32, 32)
            result["jet_constituents"] = j["constituents"]  # (n_const, 4): [pt, y, phi, pdgid]

        results.append(result)

    return results, n_total


# ──────────────────────────────────────────────────────────────────────
# Herwig batch runner (runs in a separate thread)
# ──────────────────────────────────────────────────────────────────────

def find_root_file(workdir, seed, run_name):
    """
    Find the ROOT file produced by Herwig for a given seed.
    Tries common naming patterns: {run_name}-S{seed}.root, {run_name}-s{seed}.root
    """
    patterns = [
        os.path.join(workdir, f"{run_name}-S{seed}.root"),
        os.path.join(workdir, f"{run_name}-s{seed}.root"),
    ]
    for p in patterns:
        if os.path.isfile(p):
            return p

    # Fallback: glob for anything with this seed
    matches = glob.glob(os.path.join(workdir, f"{run_name}*{seed}*.root"))
    if matches:
        return matches[0]

    return None


def herwig_runner(file_queue, run_file, workdir, n_batches, batch_size,
                  seed_start, run_name):
    """
    Thread target: run Herwig in batches and put ROOT file paths in the queue.
    """
    # Restore the original LD_LIBRARY_PATH and PATH for the Herwig subprocess
    herwig_env = os.environ.copy()
    herwig_env['LD_LIBRARY_PATH'] = os.environ.get(_HERWIG_ENV_KEY, '')
    herwig_env['PATH'] = os.environ.get('_HERWIG_ORIG_PATH', herwig_env.get('PATH', ''))

    for i in range(n_batches):
        seed = seed_start + i
        print(f"\n[Herwig] Starting batch {i+1}/{n_batches} (seed={seed}, "
              f"N={batch_size})...")

        cmd = ["Herwig", "run", run_file, f"-N{batch_size}", f"-s{seed}"]
        try:
            result = subprocess.run(cmd, cwd=workdir, env=herwig_env,
                                    capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[Herwig] WARNING: batch {i+1} returned code {result.returncode}")
                if result.stderr:
                    print(f"[Herwig] stderr: {result.stderr[:500]}")
                continue
        except FileNotFoundError:
            print("[Herwig] ERROR: 'Herwig' command not found. Is Herwig in your PATH?")
            break

        # Find the ROOT file
        root_file = find_root_file(workdir, seed, run_name)
        if root_file is None:
            print(f"[Herwig] WARNING: could not find ROOT file for seed={seed}")
            # List what's there for debugging
            root_files = glob.glob(os.path.join(workdir, "*.root"))
            print(f"[Herwig]   ROOT files in {workdir}: {root_files}")
            continue

        print(f"[Herwig] Batch {i+1} complete: {root_file}")
        file_queue.put(root_file)

    # Sentinel to signal completion
    file_queue.put(None)


# ──────────────────────────────────────────────────────────────────────
# Results accumulator and summary
# ──────────────────────────────────────────────────────────────────────

def print_summary(all_results):
    """Print final summary statistics."""
    if not all_results:
        print("\nNo events were processed.")
        return

    n_events = len(all_results)
    truths = np.array([r["jet_truth"] for r in all_results])
    probs = np.array([r["jet_prob"] for r in all_results])
    preds = (probs > 0.5).astype(int)

    # Only consider jets with valid truth
    valid = truths >= 0
    n_valid = valid.sum()

    n_unknown = (truths == -1).sum()

    print("\n" + "=" * 55)
    print("  HERWIG + HAILO Z+JET CLASSIFICATION SUMMARY")
    print("=" * 55)
    print(f"Total events processed:        {n_events}")
    print(f"Jets with valid truth match:   {n_valid}")
    if n_unknown > 0:
        # Diagnose unknown jets: unmatched (pdgid=0) vs matched-but-unknown
        unmatched_pdgids = [r["jet_pdgid"] for r in all_results if r["jet_truth"] == -1]
        n_no_match = sum(1 for p in unmatched_pdgids if p == 0)
        n_strange = sum(1 for p in unmatched_pdgids if p != 0)
        print(f"  Unknown truth jets:          {n_unknown}")
        print(f"    No parton within dR:       {n_no_match}")
        if n_strange > 0:
            from collections import Counter
            strange = Counter(unmatched_pdgids)
            strange.pop(0, None)
            print(f"    Matched but unknown pdg:   {n_strange}  {dict(strange)}")

    if n_valid == 0:
        print("No jets with valid parton truth matching.")
        return

    # Accuracy
    jet_acc = (preds[valid] == truths[valid]).mean()

    quark_mask = truths[valid] == 1
    gluon_mask = truths[valid] == 0
    quark_eff = (preds[valid][quark_mask] == 1).mean() if quark_mask.any() else 0.0
    gluon_eff = (preds[valid][gluon_mask] == 0).mean() if gluon_mask.any() else 0.0

    print(f"\nResults ({n_valid} jets):")
    print(f"  Accuracy:         {jet_acc:.4f}")
    print(f"  Quark efficiency: {quark_eff:.4f}  ({quark_mask.sum()} quarks)")
    print(f"  Gluon efficiency: {gluon_eff:.4f}  ({gluon_mask.sum()} gluons)")

    # Confusion matrix: quark vs gluon
    confusion = np.zeros((2, 2), dtype=int)
    for r in all_results:
        if r["jet_truth"] < 0:
            continue
        t = r["jet_truth"]        # 0=gluon, 1=quark
        p = 1 if r["jet_prob"] > 0.5 else 0
        confusion[t, p] += 1

    cat_names = ["gluon", "quark"]
    print(f"\nConfusion matrix:")
    print(f"{'':>14s}  Pred: {'  '.join(f'{c:>7s}' for c in cat_names)}")
    for i, name in enumerate(cat_names):
        row = "  ".join(f"{confusion[i, j]:7d}" for j in range(2))
        print(f"  True {name:>5s}:  {row}")

    # Mean quark probability by truth
    quark_probs_valid = probs[valid]
    truths_valid = truths[valid]
    if quark_mask.any():
        print(f"\nMean quark prob (true quark): {quark_probs_valid[quark_mask].mean():.4f}")
    if gluon_mask.any():
        print(f"Mean quark prob (true gluon): {quark_probs_valid[gluon_mask].mean():.4f}")

    # Parton flavour breakdown
    from collections import Counter
    all_pdgids = [r["jet_pdgid"] for r in all_results if r["jet_truth"] >= 0]
    pdg_counts = Counter(all_pdgids)
    print(f"\nParton flavour breakdown (matched jets):")
    for pid, count in sorted(pdg_counts.items(), key=lambda x: -x[1]):
        print(f"  pdgid={pid:>3d}: {count}")

    print("=" * 55)


def save_results(all_results, filepath):
    """Save per-event results to a .npz file."""
    if not all_results:
        return
    np.savez_compressed(
        filepath,
        jet_pt=np.array([r["jet_pt"] for r in all_results]),
        jet_y=np.array([r["jet_y"] for r in all_results]),
        jet_phi=np.array([r["jet_phi"] for r in all_results]),
        jet_prob=np.array([r["jet_prob"] for r in all_results]),
        jet_truth=np.array([r["jet_truth"] for r in all_results]),
        jet_pdgid=np.array([r["jet_pdgid"] for r in all_results]),
        weight=np.array([r["weight"] for r in all_results]),
    )
    print(f"Results saved to {filepath}")


# ──────────────────────────────────────────────────────────────────────
# Interactive batch jet viewer (requires matplotlib)
# ──────────────────────────────────────────────────────────────────────

def _plot_3ch_towers(fig, gs, img_3ch, R=0.4):
    """
    Plot 3 channels of a jet image as 3D tower plots.
    img_3ch: shape (3, npix, npix) in NCHW format.
    gs: list of 3 gridspec cells for the 3 channels.
    """
    channel_names = [r"Positive ($q=+1$)", r"Negative ($q=-1$)", r"Neutral ($q=0$)"]
    channel_colors = ["#d62728", "#1f77b4", "#2ca02c"]

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

    for ch in range(3):
        ax = fig.add_subplot(gs[ch], projection="3d")
        dz = img_3ch[ch].ravel()
        mask = dz > 0
        if mask.any():
            ax.bar3d(x0[mask], y0[mask], z0[mask], dx, dy, dz[mask],
                     shade=True, color=channel_colors[ch], alpha=0.8)
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_xlabel(r"$\Delta y$", fontsize=8)
        ax.set_ylabel(r"$\Delta \phi$", fontsize=8)
        ax.set_zlabel(r"$p_T$ frac", fontsize=8)
        ax.set_title(channel_names[ch], fontsize=10)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_tick_params(labelsize=6)
        ax.view_init(elev=25, azim=-60)


def _plot_combined_towers(fig, gs_cell, img_3ch, R=0.4):
    """Plot all 3 channels overlaid in a single 3D plot.
    gs_cell: a single gridspec cell (may span multiple rows/cols).
    """
    channel_colors = ["#d62728", "#1f77b4", "#2ca02c"]
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

    ax = fig.add_subplot(gs_cell, projection="3d")
    for ch in range(3):
        dz = img_3ch[ch].ravel()
        mask = dz > 0
        if mask.any():
            ax.bar3d(x0[mask], y0[mask], z0[mask], dx, dy, dz[mask],
                     shade=True, color=channel_colors[ch], alpha=0.6,
                     label=channel_labels[ch])
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_xlabel(r"$\Delta y$", fontsize=9)
    ax.set_ylabel(r"$\Delta \phi$", fontsize=9)
    ax.set_zlabel(r"$p_T$ fraction", fontsize=9)
    ax.set_title("All channels combined", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_tick_params(labelsize=7)
    ax.view_init(elev=25, azim=-60)


def _plot_particle_spray(fig, gs_cell, constituents, jet_y, jet_phi, R=0.4):
    """
    Draw the jet as a 2D scatter of constituent particles, colored by charge.

    Parameters
    ----------
    constituents : ndarray (n_const, 4) with columns [pt, y, phi, pdgid]
    jet_y, jet_phi : jet axis for computing relative coordinates
    R : jet radius (used for axis limits)
    """
    from load_jets import PDG_TO_CHARGE

    pt = constituents[:, 0]
    y = constituents[:, 1]
    phi = constituents[:, 2]
    pdgid = constituents[:, 3].astype(int)

    # Relative coordinates centred on jet axis
    dy = y - jet_y
    dphi = phi - jet_phi
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))  # wrap to [-pi, pi)

    # Charge per particle
    charges = np.array([PDG_TO_CHARGE.get(int(pid), 0) for pid in pdgid])

    # Marker sizes proportional to pT (scale so typical particles are visible)
    pt_norm = pt / pt.max() if pt.max() > 0 else pt
    sizes = 10 + 200 * pt_norm  # range ~10 to ~210

    # Color by charge: red=+1, blue=-1, green=0
    colors = []
    for q in charges:
        if q > 0:
            colors.append("#d62728")   # red
        elif q < 0:
            colors.append("#1f77b4")   # blue
        else:
            colors.append("#2ca02c")   # green

    ax = fig.add_subplot(gs_cell)
    ax.scatter(dy, dphi, s=sizes, c=colors, alpha=0.7, edgecolors="k",
               linewidths=0.3, zorder=2)

    # Draw jet cone circle
    theta = np.linspace(0, 2 * np.pi, 80)
    ax.plot(R * np.cos(theta), R * np.sin(theta), 'k--', lw=0.8, alpha=0.5)

    # Jet axis marker
    ax.plot(0, 0, 'k+', ms=10, mew=1.5, zorder=3)

    ax.set_xlim(-R * 1.3, R * 1.3)
    ax.set_ylim(-R * 1.3, R * 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta y$", fontsize=9)
    ax.set_ylabel(r"$\Delta \phi$", fontsize=9)
    ax.set_title("Particle spray", fontsize=10)
    ax.tick_params(labelsize=7)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#d62728",
               markersize=7, label="$q=+1$"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#1f77b4",
               markersize=7, label="$q=-1$"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#2ca02c",
               markersize=7, label="$q=0$"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")


class BatchJetViewer:
    """
    Persistent interactive viewer for batch-processed jets.

    The viewer window stays open across batches.  New batches are pushed
    via ``add_batch()`` and become navigable immediately.

    Navigation:
      Left / Right  or  P / N : previous / next jet within current batch
      B             or  >>    : next batch (prints a message if unavailable)
      V             or  <<    : previous batch
      Q                       : quit (close the window)
    """

    # ── construction ──────────────────────────────────────────────────

    def __init__(self, first_results, first_label="Batch 1"):
        """
        Parameters
        ----------
        first_results : list of dicts from process_root_file (must have 'jet_image')
        first_label   : e.g. "Batch 1/10"
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        # Suppress known Axes3D _button_release bug
        import mpl_toolkits.mplot3d.axes3d as _axes3d
        _orig = _axes3d.Axes3D._button_release
        def _safe(self_ax, event):
            try:
                return _orig(self_ax, event)
            except AttributeError:
                pass
        _axes3d.Axes3D._button_release = _safe

        self.plt = plt
        self.Button = Button

        # Batch storage: list of (results, label, stats)
        self._batches = []
        self.batch_idx = 0
        self.jet_idx = 0

        # Build the figure (once)
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title(
            "Herwig + Hailo Jet Viewer"
        )

        # Buttons: << Batch | < Prev | Next > | Batch >>
        ax_pb = self.fig.add_axes([0.05, 0.01, 0.14, 0.04])
        ax_pj = self.fig.add_axes([0.24, 0.01, 0.12, 0.04])
        ax_nj = self.fig.add_axes([0.64, 0.01, 0.12, 0.04])
        ax_nb = self.fig.add_axes([0.81, 0.01, 0.14, 0.04])

        self.btn_prev_batch = Button(ax_pb, "<< Batch (V)")
        self.btn_prev_jet   = Button(ax_pj, "< Prev")
        self.btn_next_jet   = Button(ax_nj, "Next >")
        self.btn_next_batch = Button(ax_nb, "Batch >> (B)")

        self.btn_prev_batch.on_clicked(self.prev_batch)
        self.btn_prev_jet.on_clicked(self.prev_jet)
        self.btn_next_jet.on_clicked(self.next_jet)
        self.btn_next_batch.on_clicked(self.next_batch)

        self._button_axes = {ax_pb, ax_pj, ax_nj, ax_nb}

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Add the first batch and draw
        self.add_batch(first_results, first_label)

    # ── batch management ──────────────────────────────────────────────

    @staticmethod
    def _compute_stats(results):
        """Pre-compute accuracy stats for one batch."""
        valid = [r for r in results if r["jet_truth"] >= 0]
        n_valid = len(valid)
        if n_valid == 0:
            return dict(n_valid=0, n_correct=0, accuracy=0.0,
                        quark_eff=0.0, gluon_eff=0.0, n_quark=0, n_gluon=0)
        n_correct = sum(1 for r in valid
                        if (r["jet_prob"] > 0.5) == (r["jet_truth"] == 1))
        n_quark = sum(1 for r in valid if r["jet_truth"] == 1)
        n_gluon = sum(1 for r in valid if r["jet_truth"] == 0)
        q_ok = sum(1 for r in valid
                   if r["jet_truth"] == 1 and r["jet_prob"] > 0.5)
        g_ok = sum(1 for r in valid
                   if r["jet_truth"] == 0 and r["jet_prob"] <= 0.5)
        return dict(
            n_valid=n_valid, n_correct=n_correct,
            accuracy=n_correct / n_valid,
            quark_eff=q_ok / n_quark if n_quark else 0.0,
            gluon_eff=g_ok / n_gluon if n_gluon else 0.0,
            n_quark=n_quark, n_gluon=n_gluon,
        )

    def _aggregate_stats(self):
        """Compute aggregate accuracy stats across all loaded batches."""
        totals = dict(n_valid=0, n_correct=0, n_quark=0, n_gluon=0,
                      q_ok=0, g_ok=0)
        for _, _, s in self._batches:
            totals["n_valid"] += s["n_valid"]
            totals["n_correct"] += s["n_correct"]
            totals["n_quark"] += s["n_quark"]
            totals["n_gluon"] += s["n_gluon"]
            # Reconstruct absolute counts from efficiencies
            totals["q_ok"] += round(s["quark_eff"] * s["n_quark"])
            totals["g_ok"] += round(s["gluon_eff"] * s["n_gluon"])
        nv = totals["n_valid"]
        return dict(
            n_valid=nv,
            n_correct=totals["n_correct"],
            accuracy=totals["n_correct"] / nv if nv else 0.0,
            n_quark=totals["n_quark"],
            n_gluon=totals["n_gluon"],
            quark_eff=totals["q_ok"] / totals["n_quark"] if totals["n_quark"] else 0.0,
            gluon_eff=totals["g_ok"] / totals["n_gluon"] if totals["n_gluon"] else 0.0,
        )

    def add_batch(self, results, label):
        """Push a new batch of results into the viewer."""
        stats = self._compute_stats(results)
        self._batches.append((results, label, stats))
        n = len(self._batches)
        if n == 1:
            # First batch — draw it
            self.batch_idx = 0
            self.jet_idx = 0
            self.draw_jet()
        else:
            print(f"  [Viewer] {label} ready — "
                  f"press B or >> to view ({len(results)} jets)")

    @property
    def _cur(self):
        """Current (results, label, stats) tuple."""
        return self._batches[self.batch_idx]

    # ── drawing ───────────────────────────────────────────────────────

    def draw_jet(self):
        """Redraw the display for the current batch + jet."""
        # Clear all axes except buttons
        for ax in self.fig.get_axes():
            if ax not in self._button_axes:
                ax.remove()

        results, label, stats = self._cur
        r = results[self.jet_idx]

        true_label = ("Quark" if r["jet_truth"] == 1 else
                      "Gluon" if r["jet_truth"] == 0 else "Unknown")
        pred_label = "Quark" if r["jet_prob"] > 0.5 else "Gluon"
        prob = r["jet_prob"]

        if r["jet_truth"] >= 0:
            correct = (true_label == pred_label)
            status = "CORRECT" if correct else "WRONG"
            status_color = "green" if correct else "red"
        else:
            status = "NO TRUTH"
            status_color = "gray"

        s = stats
        acc_str = (f"Accuracy: {s['n_correct']}/{s['n_valid']} "
                   f"({s['accuracy']:.1%})   "
                   f"Q: {s['quark_eff']:.1%} ({s['n_quark']})   "
                   f"G: {s['gluon_eff']:.1%} ({s['n_gluon']})")

        self.fig.suptitle(
            f"{label}  [{self.batch_idx+1}/{len(self._batches)} loaded]"
            f"   |   {acc_str}\n"
            f"Jet {self.jet_idx + 1}/{len(results)}   |   "
            f"$p_T$={r['jet_pt']:.0f} GeV, y={r['jet_y']:.2f}   |   "
            f"True: {true_label}   |   "
            f"Predicted: {pred_label} (prob={prob:.3f})   |   "
            f"{status}",
            fontsize=11, fontweight="bold", color=status_color,
            y=0.98,
        )

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 4, figure=self.fig,
                      top=0.84, bottom=0.08, hspace=0.3, wspace=0.35)

        img = r["jet_image"]
        _plot_3ch_towers(self.fig, [gs[0, 0], gs[0, 1], gs[0, 2]], img, R=0.4)
        _plot_combined_towers(self.fig, gs[1, 0:3], img, R=0.4)
        _plot_particle_spray(self.fig, gs[:, 3], r["jet_constituents"],
                             r["jet_y"], r["jet_phi"], R=0.4)

        # ── aggregate stats (all loaded batches) in lower-left ────────
        agg = self._aggregate_stats()
        agg_lines = (
            f"  AGGREGATE ({len(self._batches)} batch"
            f"{'es' if len(self._batches) != 1 else ''})\n\n"
            f"  Jets:       {agg['n_valid']}  "
            f"({agg['n_quark']}q + {agg['n_gluon']}g)\n"
            f"  Accuracy:   {agg['n_correct']}/{agg['n_valid']} "
            f"({agg['accuracy']:.1%})\n"
            f"  Quark eff:  {agg['quark_eff']:.1%}\n"
            f"  Gluon eff:  {agg['gluon_eff']:.1%}"
        )
        self.fig.text(
            0.005, 0.30, agg_lines,
            fontsize=12, fontfamily="monospace", fontweight="bold",
            va="center",
            bbox=dict(boxstyle="round,pad=0.7", fc="#2c3e50",
                      ec="#1a252f", alpha=0.92, lw=2),
            color="#ecf0f1",
            transform=self.fig.transFigure,
        )

        self.fig.canvas.draw_idle()

    # ── on-screen message overlay ────────────────────────────────────

    def _show_message(self, msg, color="red"):
        """Flash a temporary text overlay in the centre of the figure."""
        # Remove any previous message
        self._clear_message()
        self._msg_text = self.fig.text(
            0.5, 0.5, msg,
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.6", fc=color, alpha=0.85),
            transform=self.fig.transFigure, zorder=100,
        )
        self.fig.canvas.draw_idle()
        # Schedule removal after ~1.5 s via a one-shot timer
        self._msg_timer = self.fig.canvas.new_timer(interval=3000)
        self._msg_timer.add_callback(self._clear_message)
        self._msg_timer.single_shot = True
        self._msg_timer.start()

    def _clear_message(self):
        """Remove the overlay text if present."""
        txt = getattr(self, "_msg_text", None)
        if txt is not None:
            try:
                txt.remove()
            except ValueError:
                pass
            self._msg_text = None
            self.fig.canvas.draw_idle()

    # ── navigation callbacks ──────────────────────────────────────────

    def next_jet(self, event=None):
        results = self._cur[0]
        self.jet_idx = (self.jet_idx + 1) % len(results)
        self.draw_jet()

    def prev_jet(self, event=None):
        results = self._cur[0]
        self.jet_idx = (self.jet_idx - 1) % len(results)
        self.draw_jet()

    def next_batch(self, event=None):
        if self.batch_idx + 1 < len(self._batches):
            self.batch_idx += 1
            self.jet_idx = 0
            self.draw_jet()
        else:
            self._show_message(
                "Next batch not available yet\n— still being generated —",
                color="#c0392b")

    def prev_batch(self, event=None):
        if self.batch_idx > 0:
            self.batch_idx -= 1
            self.jet_idx = 0
            self.draw_jet()
        else:
            self._show_message(
                "Already at the first batch",
                color="#7f8c8d")

    def on_key(self, event):
        if event.key in ("right", "n"):
            self.next_jet()
        elif event.key in ("left", "p"):
            self.prev_jet()
        elif event.key == "b":
            self.next_batch()
        elif event.key in ("v", "B"):       # V or Shift+B
            self.prev_batch()
        elif event.key == "q":
            self.plt.close(self.fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Concurrent Herwig + Hailo Z+jet classification driver"
    )

    # Model
    parser.add_argument("--tag", type=str, required=True,
                        help="Model variant tag (e.g. 3ch_16-32-64)")
    parser.add_argument("--outdir", type=str, default="output",
                        help="Directory containing HEF files (default: output)")
    parser.add_argument("--hef", type=str, default=None,
                        help="Direct path to HEF file (overrides tag-based path)")

    # Herwig
    parser.add_argument("--run-file", type=str, default="LHC-Zjet.run",
                        help="Herwig .run file name (default: LHC-Zjet.run)")
    parser.add_argument("--workdir", type=str, default="Herwig",
                        help="Herwig working directory (default: Herwig)")
    parser.add_argument("--n-batches", type=int, default=10,
                        help="Number of Herwig batches to run (default: 10)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Events per Herwig batch (default: 1000)")
    parser.add_argument("--seed-start", type=int, default=1,
                        help="Starting random seed (default: 1)")

    # Jet finding and fiducial cuts
    parser.add_argument("--jet-R", type=float, default=0.4,
                        help="Jet radius for anti-kT (default: 0.4)")
    parser.add_argument("--jet-pt-min", type=float, default=500.0,
                        help="Minimum jet pT in GeV (default: 500.0)")
    parser.add_argument("--jet-pt-max", type=float, default=550.0,
                        help="Maximum jet pT in GeV (default: 550.0)")
    parser.add_argument("--jet-y-max", type=float, default=1.7,
                        help="Maximum jet |y| (default: 1.7)")

    # Output
    parser.add_argument("--results", type=str, default=None,
                        help="Save per-event results to .npz file (optional)")

    # Display
    parser.add_argument("--show", action="store_true", default=False,
                        help="Show interactive 3D jet viewer after each batch")

    # Mode: skip Herwig, just process existing ROOT files
    parser.add_argument("--root-files", type=str, nargs="+", default=None,
                        help="Process existing ROOT files instead of running Herwig")

    args = parser.parse_args()

    # Resolve HEF path
    if args.hef is None:
        args.hef = os.path.join(args.outdir, f"jet_classifier_{args.tag}.hef")
    if not os.path.isfile(args.hef):
        print(f"ERROR: HEF file not found: {args.hef}")
        sys.exit(1)
    print(f"Using HEF: {args.hef}")

    # Derive the run name from the .run file (e.g. LHC-Dijets.run -> LHC-Dijets)
    run_name = os.path.splitext(os.path.basename(args.run_file))[0]

    # Enable interactive matplotlib if --show
    viewer = None
    if args.show:
        import matplotlib.pyplot as plt
        plt.ion()
        print("  Viewer:     enabled (arrows=jets, B/V=batch nav, Q=quit)")

    all_results = []

    if args.root_files:
        # ──────────────────────────────────────────────
        # Offline mode: process existing ROOT files
        # ──────────────────────────────────────────────
        print(f"\nProcessing {len(args.root_files)} existing ROOT file(s)...")
        for i, rf in enumerate(args.root_files):
            print(f"\n[Batch {i+1}/{len(args.root_files)}] Processing {rf}")
            if not os.path.isfile(rf):
                print(f"  WARNING: file not found, skipping")
                continue
            results, n_total = process_root_file(
                rf, args.hef, jet_R=args.jet_R, jet_pt_min=args.jet_pt_min,
                jet_pt_max=args.jet_pt_max, jet_y_max=args.jet_y_max,
                show=args.show
            )
            print(f"  Events with >=1 jet: {len(results)}/{n_total}")
            if results:
                n_valid = sum(1 for r in results if r["jet_truth"] >= 0)
                print(f"  Jets with valid truth: {n_valid}")
            all_results.extend(results)

            # Push batch to persistent viewer
            if args.show and results:
                batch_label = f"Batch {i+1}/{len(args.root_files)}"
                if viewer is None:
                    viewer = BatchJetViewer(results, first_label=batch_label)
                else:
                    viewer.add_batch(results, batch_label)

    else:
        # ──────────────────────────────────────────────
        # Live mode: run Herwig concurrently with Hailo
        # ──────────────────────────────────────────────
        if not os.path.isfile(os.path.join(args.workdir, args.run_file)):
            print(f"ERROR: Run file not found: {os.path.join(args.workdir, args.run_file)}")
            print("Have you run 'Herwig read LHC-Zjet.in'?")
            sys.exit(1)

        print(f"\nStarting concurrent Herwig + Hailo pipeline:")
        print(f"  Batches:    {args.n_batches} x {args.batch_size} events")
        print(f"  Seeds:      {args.seed_start} to {args.seed_start + args.n_batches - 1}")
        print(f"  Jet algo:   anti-kT R={args.jet_R}")
        print(f"  Jet cuts:   pT in [{args.jet_pt_min}, {args.jet_pt_max}] GeV, |y| < {args.jet_y_max}")
        print(f"  Workdir:    {args.workdir}")

        file_queue = queue.Queue()

        # Start Herwig thread
        herwig_thread = threading.Thread(
            target=herwig_runner,
            args=(file_queue, args.run_file, args.workdir,
                  args.n_batches, args.batch_size, args.seed_start, run_name),
            daemon=True,
        )
        herwig_thread.start()

        # Process ROOT files as they arrive.
        # When --show is active, use non-blocking queue.get so we can
        # keep the matplotlib event loop alive between batches.
        batch_num = 0
        herwig_done = False
        while not herwig_done:
            # ── get next ROOT file ──
            if args.show:
                # Non-blocking poll: keep GUI responsive while waiting
                root_file = None
                while root_file is None and not herwig_done:
                    try:
                        root_file = file_queue.get(timeout=0.15)
                    except queue.Empty:
                        # Pump matplotlib events so viewer stays interactive
                        if viewer is not None:
                            viewer.fig.canvas.flush_events()
                        continue
            else:
                root_file = file_queue.get()

            if root_file is None:
                herwig_done = True
                break

            batch_num += 1
            print(f"\n[Hailo] Processing batch {batch_num}: {root_file}")
            t0 = time.time()

            results, n_total = process_root_file(
                root_file, args.hef, jet_R=args.jet_R,
                jet_pt_min=args.jet_pt_min, jet_pt_max=args.jet_pt_max,
                jet_y_max=args.jet_y_max, show=args.show
            )

            elapsed = time.time() - t0
            print(f"  Events with >=1 jet: {len(results)}/{n_total}")
            if results:
                # Quick per-batch accuracy
                valid = [r for r in results if r["jet_truth"] >= 0]
                if valid:
                    truths = [r["jet_truth"] for r in valid]
                    preds = [1 if r["jet_prob"] > 0.5 else 0 for r in valid]
                    acc = np.mean(np.array(preds) == np.array(truths))
                    print(f"  Accuracy: {acc:.4f} ({len(truths)} jets)")
            print(f"  Processing time: {elapsed:.1f}s")

            all_results.extend(results)

            # Push batch to persistent viewer
            if args.show and results:
                batch_label = f"Batch {batch_num}/{args.n_batches}"
                if viewer is None:
                    viewer = BatchJetViewer(results, first_label=batch_label)
                else:
                    viewer.add_batch(results, batch_label)

        herwig_thread.join()

    # ──────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────
    print_summary(all_results)

    if args.results:
        save_results(all_results, args.results)

    # Keep the viewer open until the user closes it
    if viewer is not None:
        print("\n[Viewer] All batches loaded. Close the window or press Q to exit.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
