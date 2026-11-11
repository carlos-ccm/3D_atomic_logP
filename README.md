# Machine Learning for Mapping 3D Atomic Contributions to the n-octanol/water logP from Quantum-Mechanical Solvation Calculations

**Authors:** Carlos Cruz-Marín, Albert Herrero, Enric Herrero, Carles Curutchet, F. Javier Luque, J. Vázquez
**Organization:** Universitat de Barcelona
**Year:** 2026

---

## Overview
This repository provides tools for **predicting atomic contributions to the n-octanol/water logP** using a **Graph Neural Network (GNN)** trained on quantum-mechanical solvation calculations.

It contains two main scripts:

- `train.py` — trains the GNN on atomic logP values
- `predict.py` — predicts atomic logP contributions for new molecules

---

## Installation

```bash
git clone https://github.com/carlos-ccm/3D_atomic_logP.git
cd 3D_atomic_logP
conda env create -f environment.yml
conda activate atomic_logp_gnn
```

Main dependencies: Python 3.8, RDKit 2024.03.3, PyTorch 2.1.2, PyTorch Geometric 2.6.1, scikit-learn, pandas, tqdm.

The scripts run on CPU or on a CUDA GPU (detected automatically).

---

## Input requirements

- Format: `.sdf` or gzipped `.sdf.gz`.
- Molecules must have **3D coordinates** and **explicit hydrogens** (the model predicts a contribution for every atom, hydrogens included). `predict.py` stops if a molecule lacks explicit hydrogens.

## `train.py` — Model Training

Trains the GNN to predict atomic logP.

### Usage
```bash
python train.py -i <training_set.sdf>
```

### Arguments
| Flag | Description |
|------|-------------|
| `-i <input.sdf>` | Training set (`.sdf`). See *Input requirements*. |


### Outputs
| Path | Content |
|------|---------|
| `models/atomic_gnn.pth`, `models/mu.pt`, `models/sd.pt` | Trained model |
| `preds/train_predictions.csv` | Per-atom true vs. predicted atomic logP for the training set |
| `preds/val_predictions.csv` | Per-atom true vs. predicted atomic logP for the validation set |

---

## `predict.py` — Prediction

Predicts the atomic logP contributions of every molecule in an SDF file.

### Usage
```bash
python predict.py -i <input.sdf> -o <output.sdf> --model_dir <path/to/model_dir>
```

### Arguments
| Flag | Description |
|------|-------------|
| `-i <input.sdf>` | Molecules to predict (`.sdf` or `.sdf.gz`). See *Input requirements*. |
| `-o <output.sdf>` | Output SDF with the predictions. |
| `--model_dir <dir>` | Directory with `atomic_gnn.pth`, `mu.pt` and `sd.pt` (default: `models/`). |
| `--num_workers <n>` | Number of CPU processes used to build the molecular graphs (default: 8). |
| `--molxplore` | Optional flag. Appends an extra column to the per-atom values so the file can be loaded in Pharmacelera's MolXplore viewer. |

### Output
The output SDF keeps the input molecules and their properties, and adds:

- Per atom, the `molFileValue` property (SDF `V` lines) where the predicted atomic logP is in the **second column** (the same position it is read from for training).
- Per molecule, the `ML_molecular_logP` property: the sum of the atomic logP contributions.

---

## Repository structure

```
train.py              Training script
predict.py            Prediction script
src/model.py          AtomicGNN architecture (default arguments = hyperparameters of the paper)
src/training.py       Training loop and per-element normalization of the targets
src/graph.py          Molecule → graph conversion
src/descriptors.py    3D environment descriptors
src/hydrogen_bond.py  Hydrogen-bond acceptors/donors and intramolecular H-bonds
src/geometry.py       Geometric and aromaticity helpers
src/logger.py         Console messages
environment.yml       Conda environment
```

## Citation
If you use this tool in your research, please cite:

> C. Cruz-Marín, A. Herrero, E. Herrero, C. Curutchet, F. J. Luque, J. Vázquez.
> *Machine Learning for Mapping 3D Atomic Contributions to the n-octanol/water logP from Quantum-Mechanical Solvation Calculations.*