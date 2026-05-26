# EEG Decoding with EEGNet: Subject-Dependent, Subject-Independent and Transfer Learning Approaches

## Overview

This project investigates the impact of personalization in EEG decoding using convolutional neural networks.
Three training strategies are compared:

* Subject-Dependent (SD)
* Subject-Independent (SI)
* Transfer Learning (TL)

The implementation is based on EEGNet and evaluates performance on EEG recordings collected during cognitive memory tasks.

---

## Project Structure

```text
project/
│
├── data/
│   └── *.npy
│
├── experiments/
│   ├── experiments_eegnet.py
│   └── run_experiments.py
│
├── utils/
│   └── train_utils.py
│
├── plots/
│   └── plots.py
│
├── results/
│   └── eegnet/
│
└── thesis/
```

---

## Dataset Format

Each `.npy` file contains:

```python
{
    "X": EEG trials tensor,
    "y": class labels
}
```

Tensor shape:

```python
(trials, channels, time_samples)
```

Classes:

* 1 → Visual
* 3 → Spatial
* 5 → Verbal

---

## Experimental Strategies

### Subject-Dependent (SD)

Train and test on the same subject using stratified train/test splits.

### Subject-Independent (SI)

Leave-one-subject-out training:

* train on all other subjects
* test on unseen subject

### Transfer Learning (TL)

Two-stage procedure:

1. pretraining on all other subjects
2. fine-tuning on a subset of target-subject trials

---

## Running Experiments

Run all experiments:

```bash
python experiments/run_experiments.py
```

---

## Generating Figures and Tables

```bash
python plots/plots.py
```

Generated outputs:

* violin plots
* confusion matrices
* paired subject comparisons
* LaTeX statistical tables

---

## Main Research Questions

* Does personalization improve EEG decoding performance?
* How does transfer learning compare to fully subject-dependent training?
* How much fine-tuning data is necessary for TL to become useful?
* What is the effect of pseudo-trial smoothing?
* Can smoothed EEG signals support real-time prediction?

---

## Requirements

Main libraries:

* PyTorch
* NumPy
* Scikit-learn
* Pandas
* Matplotlib
* Seaborn
* SciPy

Install dependencies:

```bash
pip install -r requirements.txt
```
