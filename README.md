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

load_jets.py                Data loading + 3-channel image conversion
model.py                    CNN architecture definition
train.py                    Training pipeline
export_onnx.py              ONNX export
hailo_convert.py            Full Hailo conversion (parse + quantise + compile)
hailo_compile.py            Hailo compile only (Stage 3)
hailo_infer.py              Hailo-8 inference on RPi
hailo_infer_show.py         Interactive viewer with 3D tower plots
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
