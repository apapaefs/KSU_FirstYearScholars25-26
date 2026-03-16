"""
Concurrent Herwig + Hailo di-jet classification driver.

Runs Herwig 7 in batches (each with a different seed), producing ROOT files.
As each ROOT file completes, reads it, finds jets with anti-kT R=0.4,
matches to parton-level truth, builds 3-channel jet images, and classifies
each jet using the Hailo-8.

Usage (on Raspberry Pi 5 with Hailo-8):
    PYTHONPATH=/usr/lib/python3/dist-packages python3 hailo_herwig_driver.py \
        --tag 3ch_16-32-64 \
        --run-file Herwig/LHC-Dijets.run \
        --workdir Herwig/ \
        --n-batches 10 --batch-size 1000 --seed-start 1

Requirements:
    pip install uproot fastjet awkward numpy
"""

import argparse
import glob
import os
import sys
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
# Neutrino PDG IDs to exclude from jet finding
# ──────────────────────────────────────────────────────────────────────
NEUTRINO_IDS = {12, -12, 14, -14, 16, -16}


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

    # Filter: remove neutrinos, apply pt and eta cuts
    mask = np.ones(len(particles), dtype=bool)
    for nid in NEUTRINO_IDS:
        mask &= (pdgid != nid)
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
                labels.append(-1)  # unknown (shouldn't happen for pp -> jj)
        else:
            labels.append(-1)
            matched_pdgids.append(0)

    return labels, matched_pdgids


# ──────────────────────────────────────────────────────────────────────
# Jet classification via Hailo
# ──────────────────────────────────────────────────────────────────────

def classify_jets(jet_constituents_list, hef_path, R=0.4):
    """
    Build 3-channel jet images and run Hailo inference.

    Parameters
    ----------
    jet_constituents_list : list of ndarray, each (n_const, 4) [pt, y, phi, pdgid]
    hef_path              : path to Hailo HEF file
    R                     : jet radius for image construction

    Returns
    -------
    probs : ndarray of shape (n_jets,) with quark probabilities
    """
    if len(jet_constituents_list) == 0:
        return np.array([])

    # Build images: NCHW (3, 32, 32) then transpose to NHWC (32, 32, 3) for Hailo
    images = []
    for const in jet_constituents_list:
        img = jet_to_image_3ch(const, R=R, npixels=32, pt_min=0.0, normalize=True)
        img_nhwc = np.transpose(img, (1, 2, 0))  # (3,32,32) -> (32,32,3)
        images.append(img_nhwc)

    images = np.stack(images).astype(np.float32)

    # Run Hailo inference
    probs = run_inference(hef_path, images)
    return probs


# ──────────────────────────────────────────────────────────────────────
# Process a single ROOT file
# ──────────────────────────────────────────────────────────────────────

def process_root_file(filepath, hef_path, jet_R=0.4, jet_pt_min=20.0):
    """
    Read a Herwig ROOT file, find jets, match truth, classify with Hailo.

    Returns
    -------
    results : list of dicts, one per event with >=2 jets:
        jet1_pt, jet1_y, jet1_phi, jet1_prob, jet1_truth, jet1_pdgid,
        jet2_pt, jet2_y, jet2_phi, jet2_prob, jet2_truth, jet2_pdgid,
        weight
    n_total : total number of events in the file
    """
    events = read_herwig_root(filepath)
    n_total = len(events)
    results = []

    for evt in events:
        # Find jets
        jets = find_jets(evt["particles"], R=jet_R, pt_min_jet=jet_pt_min)

        # Need at least 2 jets
        if len(jets) < 2:
            continue

        # Take the two leading jets
        j1, j2 = jets[0], jets[1]

        # Match to parton truth
        labels, pdgids = match_jets_to_partons([j1, j2], evt["partons"], dR_max=jet_R)

        # Classify both jets
        probs = classify_jets(
            [j1["constituents"], j2["constituents"]],
            hef_path, R=jet_R
        )

        results.append({
            "jet1_pt": j1["pt"], "jet1_y": j1["y"], "jet1_phi": j1["phi"],
            "jet1_prob": float(probs[0]), "jet1_truth": labels[0],
            "jet1_pdgid": pdgids[0],
            "jet2_pt": j2["pt"], "jet2_y": j2["y"], "jet2_phi": j2["phi"],
            "jet2_prob": float(probs[1]), "jet2_truth": labels[1],
            "jet2_pdgid": pdgids[1],
            "weight": evt["weight"],
        })

    return results, n_total


# ──────────────────────────────────────────────────────────────────────
# Herwig batch runner (runs in a separate thread)
# ──────────────────────────────────────────────────────────────────────

def find_root_file(workdir, seed, run_name):
    """
    Find the ROOT file produced by Herwig for a given seed.
    Tries common naming patterns: {run_name}-s{seed}.root, {run_name}-S{seed}.root
    """
    patterns = [
        os.path.join(workdir, f"{run_name}-s{seed}.root"),
        os.path.join(workdir, f"{run_name}-S{seed}.root"),
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
    for i in range(n_batches):
        seed = seed_start + i
        print(f"\n[Herwig] Starting batch {i+1}/{n_batches} (seed={seed}, "
              f"N={batch_size})...")

        cmd = ["Herwig", "run", run_file, f"-N{batch_size}", f"-s{seed}"]
        try:
            result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
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

    # Flatten
    n_events = len(all_results)
    jet1_truths = np.array([r["jet1_truth"] for r in all_results])
    jet2_truths = np.array([r["jet2_truth"] for r in all_results])
    jet1_probs = np.array([r["jet1_prob"] for r in all_results])
    jet2_probs = np.array([r["jet2_prob"] for r in all_results])
    jet1_preds = (jet1_probs > 0.5).astype(int)
    jet2_preds = (jet2_probs > 0.5).astype(int)

    # Only consider events where both jets have valid truth
    valid = (jet1_truths >= 0) & (jet2_truths >= 0)
    n_valid = valid.sum()

    print("\n" + "=" * 55)
    print("  HERWIG + HAILO DI-JET CLASSIFICATION SUMMARY")
    print("=" * 55)
    print(f"Total events processed:          {n_events}")
    print(f"Events with valid truth (both):  {n_valid}")

    if n_valid == 0:
        print("No events with valid parton truth matching.")
        return

    # Per-jet accuracy
    all_truths = np.concatenate([jet1_truths[valid], jet2_truths[valid]])
    all_preds = np.concatenate([jet1_preds[valid], jet2_preds[valid]])
    n_jets = len(all_truths)
    jet_acc = (all_preds == all_truths).mean()

    quark_mask = all_truths == 1
    gluon_mask = all_truths == 0
    quark_eff = (all_preds[quark_mask] == 1).mean() if quark_mask.any() else 0.0
    gluon_eff = (all_preds[gluon_mask] == 0).mean() if gluon_mask.any() else 0.0

    print(f"\nPer-jet results ({n_jets} jets):")
    print(f"  Accuracy:         {jet_acc:.4f}")
    print(f"  Quark efficiency: {quark_eff:.4f}  ({quark_mask.sum()} quarks)")
    print(f"  Gluon efficiency: {gluon_eff:.4f}  ({gluon_mask.sum()} gluons)")

    # Event-level: both jets correct
    both_correct = ((jet1_preds[valid] == jet1_truths[valid]) &
                    (jet2_preds[valid] == jet2_truths[valid]))
    evt_acc = both_correct.mean()
    print(f"\nEvent-level accuracy (both jets correct): {evt_acc:.4f}")

    # Event-level confusion matrix
    # True categories: qq, qg, gq, gg (ordered by jet1, jet2)
    # Predicted categories: same
    cat_names = ["qq", "qg", "gq", "gg"]

    def event_cat(l1, l2):
        if l1 == 1 and l2 == 1: return 0  # qq
        if l1 == 1 and l2 == 0: return 1  # qg
        if l1 == 0 and l2 == 1: return 2  # gq
        if l1 == 0 and l2 == 0: return 3  # gg
        return -1

    confusion = np.zeros((4, 4), dtype=int)
    for r in all_results:
        if r["jet1_truth"] < 0 or r["jet2_truth"] < 0:
            continue
        true_cat = event_cat(r["jet1_truth"], r["jet2_truth"])
        pred_cat = event_cat(
            1 if r["jet1_prob"] > 0.5 else 0,
            1 if r["jet2_prob"] > 0.5 else 0
        )
        if true_cat >= 0 and pred_cat >= 0:
            confusion[true_cat, pred_cat] += 1

    print(f"\nEvent-level confusion matrix:")
    print(f"{'':>12s}  Pred:  {'  '.join(f'{c:>5s}' for c in cat_names)}")
    for i, name in enumerate(cat_names):
        row = "  ".join(f"{confusion[i, j]:5d}" for j in range(4))
        print(f"  True {name}:  {row}")

    # Parton flavour breakdown
    jet1_pdgids = [r["jet1_pdgid"] for r in all_results if r["jet1_truth"] >= 0]
    jet2_pdgids = [r["jet2_pdgid"] for r in all_results if r["jet2_truth"] >= 0]
    all_pdgids = jet1_pdgids + jet2_pdgids
    from collections import Counter
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
        jet1_pt=np.array([r["jet1_pt"] for r in all_results]),
        jet1_y=np.array([r["jet1_y"] for r in all_results]),
        jet1_phi=np.array([r["jet1_phi"] for r in all_results]),
        jet1_prob=np.array([r["jet1_prob"] for r in all_results]),
        jet1_truth=np.array([r["jet1_truth"] for r in all_results]),
        jet1_pdgid=np.array([r["jet1_pdgid"] for r in all_results]),
        jet2_pt=np.array([r["jet2_pt"] for r in all_results]),
        jet2_y=np.array([r["jet2_y"] for r in all_results]),
        jet2_phi=np.array([r["jet2_phi"] for r in all_results]),
        jet2_prob=np.array([r["jet2_prob"] for r in all_results]),
        jet2_truth=np.array([r["jet2_truth"] for r in all_results]),
        jet2_pdgid=np.array([r["jet2_pdgid"] for r in all_results]),
        weight=np.array([r["weight"] for r in all_results]),
    )
    print(f"Results saved to {filepath}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Concurrent Herwig + Hailo di-jet classification driver"
    )

    # Model
    parser.add_argument("--tag", type=str, required=True,
                        help="Model variant tag (e.g. 3ch_16-32-64)")
    parser.add_argument("--outdir", type=str, default="output",
                        help="Directory containing HEF files (default: output)")
    parser.add_argument("--hef", type=str, default=None,
                        help="Direct path to HEF file (overrides tag-based path)")

    # Herwig
    parser.add_argument("--run-file", type=str, default="LHC-Dijets.run",
                        help="Herwig .run file name (default: LHC-Dijets.run)")
    parser.add_argument("--workdir", type=str, default="Herwig",
                        help="Herwig working directory (default: Herwig)")
    parser.add_argument("--n-batches", type=int, default=10,
                        help="Number of Herwig batches to run (default: 10)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Events per Herwig batch (default: 1000)")
    parser.add_argument("--seed-start", type=int, default=1,
                        help="Starting random seed (default: 1)")

    # Jet finding
    parser.add_argument("--jet-R", type=float, default=0.4,
                        help="Jet radius for anti-kT (default: 0.4)")
    parser.add_argument("--jet-pt-min", type=float, default=20.0,
                        help="Minimum jet pT in GeV (default: 20.0)")

    # Output
    parser.add_argument("--results", type=str, default=None,
                        help="Save per-event results to .npz file (optional)")

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
                rf, args.hef, jet_R=args.jet_R, jet_pt_min=args.jet_pt_min
            )
            print(f"  Events with >=2 jets: {len(results)}/{n_total}")
            if results:
                n_valid = sum(1 for r in results
                              if r["jet1_truth"] >= 0 and r["jet2_truth"] >= 0)
                print(f"  Events with valid truth: {n_valid}")
            all_results.extend(results)

    else:
        # ──────────────────────────────────────────────
        # Live mode: run Herwig concurrently with Hailo
        # ──────────────────────────────────────────────
        if not os.path.isfile(os.path.join(args.workdir, args.run_file)):
            print(f"ERROR: Run file not found: {os.path.join(args.workdir, args.run_file)}")
            print("Have you run 'Herwig build', integration, and 'Herwig mergegrids'?")
            sys.exit(1)

        print(f"\nStarting concurrent Herwig + Hailo pipeline:")
        print(f"  Batches:    {args.n_batches} x {args.batch_size} events")
        print(f"  Seeds:      {args.seed_start} to {args.seed_start + args.n_batches - 1}")
        print(f"  Jet algo:   anti-kT R={args.jet_R}, pT_min={args.jet_pt_min} GeV")
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

        # Process ROOT files as they arrive (main thread)
        batch_num = 0
        while True:
            root_file = file_queue.get()
            if root_file is None:
                break  # all batches done

            batch_num += 1
            print(f"\n[Hailo] Processing batch {batch_num}: {root_file}")
            t0 = time.time()

            results, n_total = process_root_file(
                root_file, args.hef,
                jet_R=args.jet_R, jet_pt_min=args.jet_pt_min
            )

            elapsed = time.time() - t0
            print(f"  Events with >=2 jets: {len(results)}/{n_total}")
            if results:
                # Quick per-batch accuracy
                valid = [r for r in results
                         if r["jet1_truth"] >= 0 and r["jet2_truth"] >= 0]
                if valid:
                    truths = ([r["jet1_truth"] for r in valid] +
                              [r["jet2_truth"] for r in valid])
                    preds = ([1 if r["jet1_prob"] > 0.5 else 0 for r in valid] +
                             [1 if r["jet2_prob"] > 0.5 else 0 for r in valid])
                    acc = np.mean(np.array(preds) == np.array(truths))
                    print(f"  Per-jet accuracy: {acc:.4f} ({len(truths)} jets)")
            print(f"  Processing time: {elapsed:.1f}s")

            all_results.extend(results)

        herwig_thread.join()

    # ──────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────
    print_summary(all_results)

    if args.results:
        save_results(all_results, args.results)


if __name__ == "__main__":
    main()
