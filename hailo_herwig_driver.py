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
                labels.append(-1)  # unknown (e.g. Z with pdgid=23)
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
    Read a Herwig ROOT file, find the leading jet, match truth, classify.

    Returns
    -------
    results : list of dicts, one per event with >=1 jet:
        jet_pt, jet_y, jet_phi, jet_prob, jet_truth, jet_pdgid, weight
    n_total : total number of events in the file
    """
    events = read_herwig_root(filepath)
    n_total = len(events)
    results = []

    for evt in events:
        # Find jets
        jets = find_jets(evt["particles"], R=jet_R, pt_min_jet=jet_pt_min)

        # Need at least 1 jet
        if len(jets) < 1:
            continue

        # Take the leading jet
        j = jets[0]

        # Match to parton truth
        labels, pdgids = match_jets_to_partons([j], evt["partons"], dR_max=jet_R)

        # Classify the jet
        probs = classify_jets([j["constituents"]], hef_path, R=jet_R)

        results.append({
            "jet_pt": j["pt"], "jet_y": j["y"], "jet_phi": j["phi"],
            "jet_prob": float(probs[0]), "jet_truth": labels[0],
            "jet_pdgid": pdgids[0],
            "weight": evt["weight"],
        })

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

    print("\n" + "=" * 55)
    print("  HERWIG + HAILO Z+JET CLASSIFICATION SUMMARY")
    print("=" * 55)
    print(f"Total events processed:        {n_events}")
    print(f"Jets with valid truth match:   {n_valid}")

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
            print(f"  Events with >=1 jet: {len(results)}/{n_total}")
            if results:
                n_valid = sum(1 for r in results if r["jet_truth"] >= 0)
                print(f"  Jets with valid truth: {n_valid}")
            all_results.extend(results)

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

        herwig_thread.join()

    # ──────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────
    print_summary(all_results)

    if args.results:
        save_results(all_results, args.results)


if __name__ == "__main__":
    main()
