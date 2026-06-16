# MCFWI

This repository contains the official implementation for the paper: **Mutually Calibrated FWI: Reconciling Structural and Semantic Consistencies in Wavefield Features**.

## 1. Overview

Deep learning full-waveform inversion (DL-FWI) reduces the inference time for seismic data processing to merely a few seconds.
Popular deep networks typically rely on shared static weights, failing to accommodate the spatial variability of wavefield propagation. Consequently, they tend to blur deep geological structures.
To overcome this limitation, a mutually calibrated FWI (MCFWI) approach driven by highly adaptive dynamic weight allocation is proposed.
Through dual-gating mechanism, MCFWI achieves bidirectional filtering of wavefield features: 1) Channel gating exploits the structural consistency of convolutions to verify the reliability of wavefield states; and 2) spatial gating leverages the semantic consistency of attention to evaluate the validity of wavefield geometry.
Comprehensive evaluations on synthetic datasets, OpenFWI and Marmousi \uppercase\expandafter{\romannumeral 2} slice, demonstrate that MCFWI achieves state-of-the-art performance. 
Compared to existing DL-FWI baseline models, our method significantly enhances the network's expressive freedom, successfully striking a balance between reconstructing a stable macro background and delineating deep geological structures. 
Furthermore, it exhibits superior macro-stratigraphic awareness and robust convergence on field data, FAN-10000m, offering a reliable and scalable solution for building high-fidelity velocity models at a moderate computational cost.
<img width="941" height="620" alt="image" src="https://github.com/user-attachments/assets/651cb1fb-5d38-4347-aca0-f14e28b0ecfe" />


## 2. Requirements & Computational Needs

This codebase requires Python 3.8+ and the following packages. You can install all dependencies via:

```bash
pip install -r requirements.txt
```

### Computational Requirements
* **OS:** Windows
* **Hardware:** NVIDIA GPU with at least 24 GB VRAM (e.g., RTX 3090 or better)
* **Training Time:** Approximately 8.35 hours on a single GPU.


## 3. Data Availability & Reproducibility

The dataset used to produce the main results in this paper is the publicly available **OpenFWI** dataset. 

Due to the large size of the dataset (minimum 20GB), it is not included directly in this repository. You can download the required data from the official OpenFWI repository/website:
* **Download Link:** https://openfwi-lanl.github.io/

**Data Setup Instructions:**
1. Download the required dataset splits from the link above.
2. Extract and place the data files into the `datasets/` directory in this project's root folder.
3. Ensure the data format matches the structure expected by the dataloader (`datasets_reader.py`).

## 4. Usage Tutorial & Quick Start

### Training the Model

To train the MCFWI model, execute the following script from the root directory:

```bash
python model_train.py
```

### Testing & Evaluation

To evaluate a trained model and reproduce the evaluation metrics, run:

```bash
python model_test.py
```

## 5. License

This project is licensed under the terms of the MIT license. See the LICENSE file for more details.
