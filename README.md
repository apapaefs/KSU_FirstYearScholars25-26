# KSU_FirstYearScholars25-26

This repository is part of a First Year Scholars project at Kennesaw State University. The aim is to train a convolutional neural network to separate between quark- and gluon-initited jets. 

# Workflow

Here is the current workflow for the repository:

1. *On KSU's TIMUR* Server: (on Andreas's MacBook pro use ```/Users/apapaefs/miniconda3/bin/python```)

```python
python train.py
```

This will generate ```training_results.pdf``` containing the training and validation loss, the validation accuracy and the ROC curve, and will save the training as ```best_jet_classifier.pt```. 

2. Export to the ONNX (Open Neural Network Exchange) format:

*On TIMUR*
   
```python
python export_onnx.py
```

This generates ```jet_classifier.onnx```

3. Convert the model to the Hailo-8 format (HEF

*On TIMUR*

First load the environment (see ```requirements.txt```).

One needs to install the "Hailo Dataflow Compiler", found on the Hailo developer's page: 

```python
pip install hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64.whl
```

```python
source ~/hailo-env/bin/activate
LD_PRELOAD=$HOME/conda-libstdcxx/lib/libstdc++.so.6 python hailo_convert.py
```

This generates: ```jet_classifier_parsed.har```, ```jet_classifier_quantized.har``` and the HEF file: ```jet_classifier.hef```. 

4. Run on the Raspberry Pi:

You can either run the inference using the ONNX format:

```python
```

or

```python
PYTHONPATH=/usr/lib/python3/dist-packages python3 hailo_infer_show.py --hef jet_classifier.hef --data QG_jets_1.npz --n 10000
```

for 10k jets. 

You can also just run the inference without the graphs using ```hailo_infer.py```.



