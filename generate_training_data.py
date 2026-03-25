#!/usr/bin/env python3
"""
Generate quark/gluon jet training data from Herwig Z+jet events.

Runs three Z+jet channels (qg, qbarg, qqbar) via Herwig, monitors output,
and stops each channel when the target number of jets is reached.
Merges the results into a single .npz file matching the Zenodo format:
    X : (N, M, 4)  — constituents [pt, y, phi, pdgid], zero-padded
    y : (N,)        — labels (0=gluon, 1=quark)

Usage:
    # 1. First prepare the .run files (once):
    #    cd Herwig && Herwig read ../LHC-Zjet-qg.in && \
    #    Herwig read ../LHC-Zjet-qbarg.in && Herwig read ../LHC-Zjet-qqbar.in
    #
    # 2. Then generate training data:
    python3 generate_training_data.py --workdir Herwig --target 50000 \
        --batch-size 5000 --output data/herwig_Zjet_50k.npz
"""

import os
import sys
import argparse
import glob
import subprocess
import time
import signal

# ── Isolate Python from Herwig's bundled libraries ──────────────────
# Herwig's activate script prepends its own lib paths (including its own
# fastjet) to LD_LIBRARY_PATH.  That clashes with the pip-installed
# fastjet C-extension.  We must clean LD_LIBRARY_PATH *before* the
# dynamic linker resolves symbols, so we re-exec if contaminated.
_HERWIG_TOKENS = ('Herwig', 'ThePEG', 'herwig', 'thepeg')

_orig_ld = os.environ.get('LD_LIBRARY_PATH', '')
if any(tok in _orig_ld for tok in _HERWIG_TOKENS) \
        and '_HERWIG_ORIG_LD_LIBRARY_PATH' not in os.environ:
    # First run with Herwig paths — save originals and re-exec clean
    os.environ['_HERWIG_ORIG_LD_LIBRARY_PATH'] = _orig_ld
    os.environ['_HERWIG_ORIG_PATH'] = os.environ.get('PATH', '')
    clean_ld = os.pathsep.join(
        p for p in _orig_ld.split(os.pathsep)
        if p and not any(tok in p for tok in _HERWIG_TOKENS)
    )
    os.environ['LD_LIBRARY_PATH'] = clean_ld
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np

import uproot
import awkward as ak
import fastjet


# ──────────────────────────────────────────────────────────────────────
# Kinematics helpers (same as hailo_herwig_driver.py)
# ──────────────────────────────────────────────────────────────────────

NEUTRINO_IDS = {12, -12, 14, -14, 16, -16}
EW_BOSON_IDS = {22, 23, 24, -24, 25}
EXCLUDE_FROM_JETS = NEUTRINO_IDS | EW_BOSON_IDS


def four_vector_to_ptyphiid(E, px, py, pz, pdgid):
    pt = np.sqrt(px**2 + py**2)
    eps = 1e-12
    y = 0.5 * np.log((E + pz + eps) / (E - pz + eps))
    phi = np.arctan2(py, px)
    return np.column_stack([pt, y, phi, pdgid])


def delta_R(y1, phi1, y2, phi2):
    dphi = phi1 - phi2
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    return np.sqrt((y1 - y2)**2 + dphi**2)


# ──────────────────────────────────────────────────────────────────────
# ROOT file reader
# ──────────────────────────────────────────────────────────────────────

def read_herwig_root(filepath):
    f = uproot.open(filepath)
    tree = f["Data"]

    numparticles = tree["numparticles"].array(library="np")
    objects_raw = tree["objects"].array(library="np")
    evweight = tree["evweight"].array(library="np")
    numoutgoing = tree["numoutgoing"].array(library="np")
    partons_raw = tree["partons"].array(library="np")

    n_events = len(numparticles)
    events = []

    for i in range(n_events):
        npart = int(numparticles[i])
        nout = int(numoutgoing[i])

        obj = objects_raw[i]
        particles = np.zeros((npart, 5), dtype=np.float64)
        particles[:, 0] = obj[0, :npart]
        particles[:, 1] = obj[1, :npart]
        particles[:, 2] = obj[2, :npart]
        particles[:, 3] = obj[3, :npart]
        particles[:, 4] = obj[4, :npart]

        par = partons_raw[i]
        partons = np.zeros((nout, 5), dtype=np.float64)
        partons[:, 0] = par[0, :nout]
        partons[:, 1] = par[1, :nout]
        partons[:, 2] = par[2, :nout]
        partons[:, 3] = par[3, :nout]
        partons[:, 4] = par[4, :nout]

        events.append({
            "particles": particles,
            "partons": partons,
            "weight": float(evweight[i]),
        })

    return events


# ──────────────────────────────────────────────────────────────────────
# Jet finding + truth matching → constituent arrays
# ──────────────────────────────────────────────────────────────────────

def find_jets(particles, R=0.4, pt_min_jet=20.0, eta_max=5.0, pt_min_part=0.1):
    E = particles[:, 0]
    px = particles[:, 1]
    py = particles[:, 2]
    pz = particles[:, 3]
    pdgid = particles[:, 4].astype(int)

    pt = np.sqrt(px**2 + py**2)
    theta = np.arctan2(pt, pz)
    eta = -np.log(np.tan(theta / 2.0 + 1e-20))

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

    pj_array = ak.zip({
        "px": filt_px, "py": filt_py, "pz": filt_pz, "E": filt_E,
    })

    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
    cluster_seq = fastjet.ClusterSequence(pj_array, jetdef)

    jets = cluster_seq.inclusive_jets(min_pt=pt_min_jet)
    const_indices = cluster_seq.constituent_index(min_pt=pt_min_jet)

    jet_px = ak.to_numpy(jets["px"])
    jet_py = ak.to_numpy(jets["py"])
    jet_pz = ak.to_numpy(jets["pz"])
    jet_E = ak.to_numpy(jets["E"])
    jet_pt = np.sqrt(jet_px**2 + jet_py**2)

    pt_order = np.argsort(-jet_pt)

    filt_ptyphiid = four_vector_to_ptyphiid(
        filt_E, filt_px, filt_py, filt_pz, filt_pdgid
    )

    jets_out = []
    for idx in pt_order:
        jpt = jet_pt[idx]
        jpx = jet_px[idx]
        jpy = jet_py[idx]
        jpz = jet_pz[idx]
        jE = jet_E[idx]

        eps = 1e-12
        jy = 0.5 * np.log((jE + jpz + eps) / (jE - jpz + eps))
        jphi = np.arctan2(jpy, jpx)

        cidx = ak.to_numpy(const_indices[idx])
        if len(cidx) == 0:
            continue

        const_arr = filt_ptyphiid[cidx]

        jets_out.append({
            "pt": float(jpt), "y": float(jy), "phi": float(jphi),
            "constituents": const_arr,  # (n_const, 4): [pt, y, phi, pdgid]
        })

    return jets_out


def match_jet_to_partons(jet, partons, dR_max=0.4):
    """Match a single jet to the closest QCD parton. Returns label (0/1/-1)."""
    if len(partons) == 0:
        return -1

    par_ptyphiid = four_vector_to_ptyphiid(
        partons[:, 0], partons[:, 1], partons[:, 2], partons[:, 3], partons[:, 4]
    )
    par_pdgid = partons[:, 4].astype(int)

    # Filter out EW bosons from parton list
    qcd_mask = np.array([abs(p) not in EW_BOSON_IDS for p in par_pdgid])
    if not qcd_mask.any():
        return -1

    best_dR = 999.0
    best_pid = 0
    for ip in range(len(partons)):
        if not qcd_mask[ip]:
            continue
        dR = delta_R(jet["y"], jet["phi"],
                     par_ptyphiid[ip, 1], par_ptyphiid[ip, 2])
        if dR < best_dR:
            best_dR = dR
            best_pid = abs(par_pdgid[ip])

    if best_dR >= dR_max:
        return -1
    if 1 <= best_pid <= 5:
        return 1  # quark
    elif best_pid == 21:
        return 0  # gluon
    return -1


# ──────────────────────────────────────────────────────────────────────
# Process a ROOT file → list of (constituents, label) tuples
# ──────────────────────────────────────────────────────────────────────

def extract_jets_from_file(filepath, jet_R=0.4, pt_min=500.0, pt_max=550.0,
                           y_max=1.7):
    """
    Returns list of (constituents_array, label) where:
        constituents_array : ndarray (n_const, 4) [pt, y, phi, pdgid]
        label              : 0=gluon, 1=quark, -1=unknown
    """
    events = read_herwig_root(filepath)
    results = []

    for evt in events:
        jets = find_jets(evt["particles"], R=jet_R, pt_min_jet=20.0)
        if not jets:
            continue

        j = jets[0]  # leading jet
        if j["pt"] < pt_min or j["pt"] > pt_max:
            continue
        if abs(j["y"]) > y_max:
            continue

        label = match_jet_to_partons(j, evt["partons"], dR_max=jet_R)
        results.append((j["constituents"], label))

    return results


# ──────────────────────────────────────────────────────────────────────
# Find ROOT file for a given seed
# ──────────────────────────────────────────────────────────────────────

def find_root_file(workdir, seed, run_name):
    patterns = [
        os.path.join(workdir, f"{run_name}-S{seed}.root"),
        os.path.join(workdir, f"{run_name}-s{seed}.root"),
    ]
    for p in patterns:
        if os.path.isfile(p):
            return p
    matches = glob.glob(os.path.join(workdir, f"{run_name}*{seed}*.root"))
    return matches[0] if matches else None


# ──────────────────────────────────────────────────────────────────────
# Channel runner: generates batches until target is met
# ──────────────────────────────────────────────────────────────────────

def run_channel(run_name, workdir, target, batch_size, seed_start,
                expected_label, jet_R, pt_min, pt_max, y_max,
                herwig_env, n_parallel=4):
    """
    Run Herwig batches for one channel until `target` jets of `expected_label`
    are collected.  Launches up to `n_parallel` Herwig processes at once.

    Parameters
    ----------
    run_name       : e.g. "LHC-Zjet-qg"
    expected_label : 1 for quark channels, 0 for gluon channel
    herwig_env     : environment dict with correct LD_LIBRARY_PATH for Herwig
    n_parallel     : number of concurrent Herwig runs (default: 4)

    Returns
    -------
    collected : list of (constituents_ndarray, label_int)
    """
    label_name = "quark" if expected_label == 1 else "gluon"
    run_file = f"{run_name}.run"

    if not os.path.isfile(os.path.join(workdir, run_file)):
        print(f"  ERROR: {os.path.join(workdir, run_file)} not found.")
        print(f"  Run: cd {workdir} && Herwig read ../{run_name}.in")
        return []

    collected = []
    n_target_label = 0
    seed = seed_start
    wave_num = 0

    print(f"\n{'='*60}")
    print(f"  Channel: {run_name} (collecting {label_name} jets)")
    print(f"  Target: {target} jets, {n_parallel} parallel runs")
    print(f"{'='*60}")

    while n_target_label < target:
        wave_num += 1

        seeds = list(range(seed, seed + n_parallel))
        print(f"\n  [{run_name}] Wave {wave_num}: launching {n_parallel} runs "
              f"(seeds {seeds[0]}–{seeds[-1]}, N={batch_size} each)...",
              flush=True)

        # Launch all Herwig processes
        procs = []
        launch_time = time.time()
        for s in seeds:
            cmd = ["Herwig", "run", run_file, f"-N{batch_size}", f"-s{s}"]
            try:
                p = subprocess.Popen(cmd, cwd=workdir, env=herwig_env,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                procs.append((s, p))
            except FileNotFoundError:
                print("  ERROR: 'Herwig' not found in PATH")
                return collected

        # Poll processes with live status display
        status = {s: "running" for s, _ in procs}
        outputs = {}  # seed -> (stdout, stderr)
        finished_seeds = []
        failed = 0

        while any(st == "running" for st in status.values()):
            for s, p in procs:
                if status[s] != "running":
                    continue
                rc = p.poll()
                if rc is not None:
                    stdout, stderr = p.communicate()
                    outputs[s] = (stdout, stderr)
                    if rc == 0:
                        status[s] = "done"
                        finished_seeds.append(s)
                    else:
                        status[s] = f"FAIL(rc={rc})"
                        failed += 1
                        if stderr:
                            print(f"\n    seed {s}: FAILED (rc={rc})")
                            print(f"      stderr: {stderr.decode()[:200]}")

            elapsed = time.time() - launch_time
            n_done = sum(1 for st in status.values() if st != "running")
            slots = []
            for s in seeds:
                st = status[s]
                if st == "running":
                    slots.append(f"s{s}:running")
                elif st == "done":
                    slots.append(f"s{s}:done")
                else:
                    slots.append(f"s{s}:FAIL")
            bar = " | ".join(slots)
            print(f"\r    [{elapsed:5.0f}s] {bar}  "
                  f"({n_done}/{n_parallel} complete)", end="", flush=True)

            if any(st == "running" for st in status.values()):
                time.sleep(2)

        elapsed = time.time() - launch_time
        print(f"\r    [{elapsed:5.0f}s] {len(finished_seeds)}/{n_parallel} "
              f"succeeded{' ' * 40}", flush=True)

        # Process ROOT files from successful runs
        wave_target = 0
        wave_other = 0
        wave_unknown = 0

        for s in finished_seeds:
            root_file = find_root_file(workdir, s, run_name)
            if root_file is None:
                print(f"    WARNING: ROOT file not found for seed={s}")
                continue

            jets = extract_jets_from_file(
                root_file, jet_R=jet_R, pt_min=pt_min, pt_max=pt_max,
                y_max=y_max
            )

            for const, label in jets:
                if label == expected_label:
                    collected.append((const, label))
                    n_target_label += 1
                    wave_target += 1
                elif label >= 0:
                    collected.append((const, label))
                    wave_other += 1
                else:
                    wave_unknown += 1

            try:
                os.remove(root_file)
            except OSError:
                pass

        print(f"    Wave {wave_num} results: {wave_target} {label_name}, "
              f"{wave_other} other, {wave_unknown} unmatched | "
              f"Total: {n_target_label}/{target}")

        seed += n_parallel

        if n_target_label >= target:
            print(f"\n  [{run_name}] Target reached: {n_target_label} "
                  f"{label_name} jets collected.")
            break

    return collected


# ──────────────────────────────────────────────────────────────────────
# Merge and convert to Zenodo .npz format
# ──────────────────────────────────────────────────────────────────────

def merge_to_zenodo_format(quark_jets, gluon_jets, n_quarks, n_gluons,
                           output_path, shuffle=True):
    """
    Merge quark and gluon jet constituent arrays into Zenodo-format .npz.

    Parameters
    ----------
    quark_jets : list of ndarray (n_const, 4) [pt, y, phi, pdgid]
    gluon_jets : list of ndarray (n_const, 4)
    n_quarks   : number of quark jets to include
    n_gluons   : number of gluon jets to include
    output_path: path for the output .npz file
    shuffle    : if True, randomly shuffle the combined dataset
    """
    quarks = quark_jets[:n_quarks]
    gluons = gluon_jets[:n_gluons]

    all_constituents = quarks + gluons
    labels = np.array([1] * len(quarks) + [0] * len(gluons), dtype=np.int32)

    # Find max multiplicity
    max_mult = max(c.shape[0] for c in all_constituents)
    n_total = len(all_constituents)

    print(f"\nBuilding dataset: {len(quarks)} quarks + {len(gluons)} gluons "
          f"= {n_total} jets")
    print(f"Max constituent multiplicity: {max_mult}")

    # Build zero-padded X array: (N, M, 4)
    X = np.zeros((n_total, max_mult, 4), dtype=np.float64)
    for i, const in enumerate(all_constituents):
        n = const.shape[0]
        X[i, :n, :] = const  # [pt, y, phi, pdgid]

    # Shuffle
    if shuffle:
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(n_total)
        X = X[perm]
        labels = labels[perm]

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, X=X, y=labels)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved to {output_path} ({file_size:.1f} MB)")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {labels.shape}")
    print(f"  Quarks: {(labels == 1).sum()}, Gluons: {(labels == 0).sum()}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate quark/gluon jet training data from Herwig Z+jet"
    )

    parser.add_argument("--workdir", type=str, default="Herwig",
                        help="Herwig working directory (default: Herwig)")
    parser.add_argument("--target", type=int, default=50000,
                        help="Target number of each jet type (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Events per Herwig batch (default: 5000)")
    parser.add_argument("--seed-start", type=int, default=1,
                        help="Starting random seed (default: 1)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Number of parallel Herwig runs per channel "
                             "(default: 4)")
    parser.add_argument("--output", type=str,
                        default="data/herwig_Zjet_50k.npz",
                        help="Output .npz file path")

    # Jet finding and fiducial cuts (match Zenodo dataset)
    parser.add_argument("--jet-R", type=float, default=0.4)
    parser.add_argument("--jet-pt-min", type=float, default=500.0)
    parser.add_argument("--jet-pt-max", type=float, default=550.0)
    parser.add_argument("--jet-y-max", type=float, default=1.7)

    # Skip generation, just merge existing collected jets
    parser.add_argument("--merge-only", type=str, nargs="+", default=None,
                        help="Merge existing .npz checkpoint files instead "
                             "of running Herwig")

    args = parser.parse_args()

    if args.merge_only:
        # Load checkpoint files and merge
        quark_jets = []
        gluon_jets = []
        for f in args.merge_only:
            data = np.load(f, allow_pickle=True)
            for const, label in zip(data["constituents"], data["labels"]):
                # Remove zero-padding
                nonzero = const[np.any(const != 0, axis=1)]
                if label == 1:
                    quark_jets.append(nonzero)
                elif label == 0:
                    gluon_jets.append(nonzero)
        print(f"Loaded {len(quark_jets)} quarks, {len(gluon_jets)} gluons "
              f"from {len(args.merge_only)} files")
        n_q = min(len(quark_jets), args.target)
        n_g = min(len(gluon_jets), args.target)
        merge_to_zenodo_format(quark_jets, gluon_jets, n_q, n_g, args.output)
        return

    # Set up Herwig environment (restore original LD_LIBRARY_PATH if stripped)
    herwig_env = os.environ.copy()
    orig_ld = os.environ.get('_HERWIG_ORIG_LD_LIBRARY_PATH', '')
    if orig_ld:
        herwig_env['LD_LIBRARY_PATH'] = orig_ld
    orig_path = os.environ.get('_HERWIG_ORIG_PATH', '')
    if orig_path:
        herwig_env['PATH'] = orig_path

    # ─── Quark channels: qg + qbarg ───
    # We need `target` quarks total from both channels combined.
    # Run each channel targeting `target` quarks; we'll trim later.
    # Use different seed ranges so they don't collide.

    channels = [
        # (run_name, expected_label, seed_offset)
        ("LHC-Zjet-qqbar", 0, 0),        # gluon channel
        ("LHC-Zjet-qg",    1, 10000),     # quark channel
        ("LHC-Zjet-qbarg", 1, 20000),     # antiquark channel (label=1=quark)
    ]

    all_quarks = []
    all_gluons = []

    t_start = time.time()

    for run_name, expected_label, seed_offset in channels:
        # How many more do we need?
        if expected_label == 1:
            remaining = args.target - len(all_quarks)
        else:
            remaining = args.target - len(all_gluons)

        if remaining <= 0:
            print(f"\n  Skipping {run_name}: already have enough "
                  f"{'quarks' if expected_label == 1 else 'gluons'}")
            continue

        collected = run_channel(
            run_name=run_name,
            workdir=args.workdir,
            target=remaining,
            batch_size=args.batch_size,
            seed_start=args.seed_start + seed_offset,
            expected_label=expected_label,
            jet_R=args.jet_R,
            pt_min=args.jet_pt_min,
            pt_max=args.jet_pt_max,
            y_max=args.jet_y_max,
            herwig_env=herwig_env,
            n_parallel=args.parallel,
        )

        for const, label in collected:
            if label == 1:
                all_quarks.append(const)
            elif label == 0:
                all_gluons.append(const)

        print(f"\n  Running totals: {len(all_quarks)} quarks, "
              f"{len(all_gluons)} gluons")

        # Save checkpoint after each channel
        ckpt = args.output.replace(".npz", f"_ckpt_{run_name}.npz")
        channel_const = [c for c, _ in collected]
        channel_labels = [l for _, l in collected]
        if channel_const:
            max_m = max(c.shape[0] for c in channel_const)
            padded = np.zeros((len(channel_const), max_m, 4))
            for i, c in enumerate(channel_const):
                padded[i, :c.shape[0], :] = c
            np.savez_compressed(
                ckpt, constituents=padded,
                labels=np.array(channel_labels, dtype=np.int32)
            )
            print(f"  Checkpoint saved: {ckpt}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Generation complete in {elapsed/60:.1f} minutes")
    print(f"  Total quarks: {len(all_quarks)}")
    print(f"  Total gluons: {len(all_gluons)}")
    print(f"{'='*60}")

    # Trim to exact target counts
    n_q = min(len(all_quarks), args.target)
    n_g = min(len(all_gluons), args.target)

    if n_q < args.target:
        print(f"WARNING: only collected {n_q}/{args.target} quark jets")
    if n_g < args.target:
        print(f"WARNING: only collected {n_g}/{args.target} gluon jets")

    merge_to_zenodo_format(all_quarks, all_gluons, n_q, n_g, args.output)


if __name__ == "__main__":
    main()
