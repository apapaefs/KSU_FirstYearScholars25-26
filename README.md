# KSU_FirstYearScholars25-26

This repository is part of a First Year Scholars project at Kennesaw State University. The aim is to train a convolutional neural network to separate between quark- and gluon-initiated jets, and deploy it on a Raspberry Pi 5 with the Hailo-8 AI HAT+.

## Project Layout

Here is the current workflow for the repository. Note that files will have a tag `_3ch_X-Y-Z`, where X, Y, and Z are the convolution layer channels.

```
data/                       Input data files
  QG_jets.npz                 Pythia jets (100k)
  QG_jets_1.npz               Pythia jets (variant)
  QG_jets_herwig_0.npz        Herwig jets

output/                     All generated files (tagged with TAG)
  best_jet_classifier_TAG.pt    PyTorch checkpoint
  jet_classifier_TAG.onnx       ONNX model
  jet_classifier_TAG.alls       Hailo quantisation config
  jet_classifier_TAG.hef        Hailo binary
  jet_classifier_TAG.info       Model metadata
  training_results_TAG.pdf      Training plots
  ...

generate_training_data.py   Herwig Z+jet training data generator
LHC-Zjet.in                Herwig input: Z+jet (all channels)
LHC-Zjet-qg.in             Herwig input: Z+jet qg channel (quark jets)
LHC-Zjet-qbarg.in          Herwig input: Z+jet qbarg channel (antiquark jets)
LHC-Zjet-qqbar.in          Herwig input: Z+jet qqbar channel (gluon jets)
LHC-Hjet.in                Herwig input: H+jet (all channels)
load_jets.py                Data loading + 3-channel image conversion
model.py                    CNN architecture definition
train.py                    Training pipeline
export_onnx.py              ONNX export
hailo_convert.py            Full Hailo conversion (parse + quantise + compile)
hailo_compile.py            Hailo compile only (Stage 3)
hailo_infer.py              Hailo-8 inference on RPi
hailo_infer_show.py         Interactive viewer with 3D tower plots
hailo_herwig_driver.py      Concurrent Herwig Z+jet generation + Hailo classification
onnx_infer.py               ONNX Runtime inference (CPU, no Hailo needed)
plot_jets.py                Simple single-jet tower plot
train_qat.py                Quantisation-aware training (optional)
workflow.tex                Full workflow documentation (LaTeX)
```

## Workflow

Every output file carries a **tag** (default: `3ch_16-32-64`) so you can train multiple model variants side by side.

### 1. Train the model

*On KSU's TIMUR server* (on Andreas's MacBook Pro use `/Users/apapaefs/miniconda3/bin/python`):

```bash
python train.py --tag 3ch_16-32-64
```

This generates in `output/`:
- `best_jet_classifier_3ch_16-32-64.pt` --- trained weights (checkpoint with architecture info)
- `training_results_3ch_16-32-64.pdf` --- loss curves, accuracy, ROC
- `jet_classifier_3ch_16-32-64.info` --- model metadata (architecture, hyperparameters, accuracy)

### 2. Export to ONNX

*On TIMUR:*

```bash
python export_onnx.py --tag 3ch_16-32-64
```

This generates `output/jet_classifier_3ch_16-32-64.onnx`. The architecture params are read automatically from the checkpoint.

### 3. Convert for the Hailo-8

*On TIMUR:*

First load the environment (see `requirements.txt`). Install the Hailo Dataflow Compiler:

```bash
pip install hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl
```

Then run:

```bash
source ~/hailo-env/bin/activate
LD_PRELOAD=$HOME/conda-libstdcxx/lib/libstdc++.so.6 python hailo_convert.py --tag 3ch_16-32-64
```

(First time: do ```conda create -p /home/apapaefs/conda-libstdcxx -c conda-forge libstdcxx-ng=13 -y```
and also: ```python3 -m venv ~/hailo-env```)

This generates in `output/`:
- `jet_classifier_3ch_16-32-64_parsed.har`
- `jet_classifier_3ch_16-32-64_quantized.har`
- `jet_classifier_3ch_16-32-64.hef`

### 4. Run on the Raspberry Pi

Using ONNX Runtime (no Hailo hardware needed):

```bash
python3 onnx_infer.py --tag 3ch_16-32-64 --data data/QG_jets.npz --n 10000
```

Using the Hailo-8:

```bash
PYTHONPATH=/usr/lib/python3/dist-packages python3 hailo_infer.py --tag 3ch_16-32-64 --data data/QG_jets.npz --n 10000
```

Interactive viewer with 3D tower plots:

```bash
PYTHONPATH=/usr/lib/python3/dist-packages python3 hailo_infer_show.py --tag 3ch_16-32-64 --data data/QG_jets_1.npz --n 10000
```

### Trying a different architecture

```bash
# Wider network: 32-64-128 filters, 256 FC hidden units
python train.py --tag 3ch_32-64-128 --c1 32 --c2 64 --c3 128 --fc 256
python export_onnx.py --tag 3ch_32-64-128
python hailo_convert.py --tag 3ch_32-64-128
```

Both sets of results coexist in `output/` without overwriting each other.

### 5. Live Herwig + Hailo classification

This mode runs Herwig 7 event generation concurrently with Hailo-8 inference on the Raspberry Pi. The driver supports any V+jet process where the boson decays invisibly (to neutrinos), leaving one hard jet per event:

- **Z+jet** (`LHC-Zjet`): Z decays to neutrinos
- **H+jet** (`LHC-Hjet`): H decays to invisible (e.g. H to neutrinos via effective coupling)

Herwig produces ROOT files in batches (each with a different seed), and the driver reads them, finds the leading jet, matches parton-level truth (excluding EW bosons), and classifies it as quark or gluon.

**Prerequisites:**

```bash
pip install uproot fastjet awkward
```

The Herwig module has to be loaded on the RPi:

```bash
module load herwig/stable
```

**Python 3.13 fix for MadGraph:** MG5_aMC v3.5.1 (bundled with Herwig) is incompatible with Python 3.13 due to PEP 667 changes to `locals()` semantics inside `exec()`. This causes `NameError: name 'mdl_lamWS' is not defined` during `Herwig build`. Apply the patch `model_reader_py313.patch` (included in this repository) to the installed MadGraph:

```bash
cd /opt/Herwig-install/opt/MG5_aMC_v3_5_1/models/
sudo cp model_reader.py model_reader.py.bak
sudo patch < /path/to/model_reader_py313.patch
```

Herwig must be set up first (choose one):

```bash
cd Herwig/
Herwig read LHC-Zjet.in    # Z+jet
Herwig read LHC-Hjet.in    # H+jet
```

**Run live (Herwig + Hailo concurrently):**

```bash
cd /path/to/project
# Z+jet example:
PYTHONPATH=/usr/lib/python3/dist-packages python3 hailo_herwig_driver.py \
    --tag 3ch_16-32-64 \
    --run-file LHC-Zjet.run \
    --workdir Herwig/ \
    --n-batches 10 --batch-size 1000 --seed-start 1

# H+jet example:
PYTHONPATH=/usr/lib/python3/dist-packages python3 hailo_herwig_driver.py \
    --tag 3ch_16-32-64 \
    --run-file LHC-Hjet.run \
    --workdir Herwig/ \
    --n-batches 10 --batch-size 1000 --seed-start 1
```

**Process existing ROOT files (no Herwig run):**

```bash
PYTHONPATH=/usr/lib/python3/dist-packages python3 hailo_herwig_driver.py \
    --tag 3ch_16-32-64 \
    --root-files Herwig/LHC-Zjet-S1.root Herwig/LHC-Zjet-S2.root
```

**With interactive viewer (3D tower plots + particle spray):**

```bash
PYTHONPATH=/usr/lib/python3/dist-packages python3 hailo_herwig_driver.py \
    --tag 3ch_16-32-64 \
    --root-files Herwig/LHC-Zjet-S1.root \
    --show
```

**Save results to file for later analysis:**

```bash
PYTHONPATH=/usr/lib/python3/dist-packages python3 hailo_herwig_driver.py \
    --tag 3ch_16-32-64 \
    --root-files Herwig/LHC-Zjet-S1.root \
    --results output/herwig_zjet_results.npz
```

### 6. Generate Herwig training data

Generate your own quark/gluon jet training dataset from Herwig Z+jet events, matching the [Zenodo dataset format](https://zenodo.org/records/3164691) (`.npz` with `X` shape `(N, M, 4)` and `y` shape `(N,)`).

Three separate Z+jet channels are used:
- `LHC-Zjet-qqbar.in` — q+qbar → Z+g → **gluon** jets
- `LHC-Zjet-qg.in` — q+g → Z+q → **quark** jets
- `LHC-Zjet-qbarg.in` — qbar+g → Z+qbar → **antiquark** jets (labelled as quark)

The generation script runs each channel in batches, monitors the jet count, and automatically stops each channel when the target is reached.

**Step 1: Prepare the .run files (once per machine):**

```bash
cd Herwig/
Herwig read ../LHC-Zjet-qqbar.in
Herwig read ../LHC-Zjet-qg.in
Herwig read ../LHC-Zjet-qbarg.in
```

**Step 2: Generate training data:**

```bash
python3 generate_training_data.py \
    --workdir Herwig \
    --target 50000 \
    --batch-size 5000 \
    --output data/herwig_Zjet_50k.npz
```

By default, 4 Herwig processes run in parallel per channel (each with a different seed). Use `--parallel` to change this:

```bash
python3 generate_training_data.py \
    --workdir Herwig \
    --target 50000 \
    --batch-size 5000 \
    --parallel 8 \
    --output data/herwig_Zjet_50k.npz
```

Use `--seed-start` to set the starting random seed (default: 1). Each channel uses a different offset added to this value (0 for qqbar, 10000 for qg, 20000 for qbarg), so seeds never collide between channels. Within each channel, seeds increment by `--parallel` per wave.

```bash
python3 generate_training_data.py \
    --workdir Herwig \
    --target 50000 \
    --batch-size 5000 \
    --seed-start 100 \
    --output data/herwig_Zjet_50k.npz
```

This produces `data/herwig_Zjet_50k.npz` containing 50k quark + 50k gluon jets (randomly shuffled), with per-jet constituent arrays `[pT, rapidity, phi, pdgid]` zero-padded to the maximum multiplicity. Fiducial cuts match the Zenodo dataset: anti-kT R=0.4, pT in [500, 550] GeV, |y| < 1.7.

Checkpoint files are saved after each channel (`*_ckpt_LHC-Zjet-*.npz`), so if the run is interrupted you can resume with:

```bash
python3 generate_training_data.py \
    --merge-only data/herwig_Zjet_50k_ckpt_*.npz \
    --output data/herwig_Zjet_50k.npz
```
