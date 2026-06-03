# Turoman LDA Baseline (Turoman et al., 2024)

Classical **linear discriminant analysis (LDA)** baseline for EEG memory decoding, replicating the methodology in Turoman et al. (2024). Also contains the **CSV → subject `.npy` conversion** used by both the LDA pipeline and the neural-network experiments.

## Overview

For each subject and temporal condition (BSL, SENSORY, DELAY):

1. Average EEG trials over time → spatial feature vectors
2. Split data with **100** stratified shuffle iterations (67% train / 33% test)
3. Build **pseudo-trials** (average of 5 trials per class, 100 pseudo-trials per class) separately in train and test
4. Center features using the training mean
5. Train shrinkage LDA (`solver="lsqr"`, `shrinkage="auto"`)
6. Report mean accuracy across iterations and row-normalized confusion matrices

## Project structure

```text
TuromanClassifier/
├── Convert_to_subject_cpy.py   # CSV → per-subject .npy (run this first)
├── TuromanClassifier.py        # LDA evaluation
├── plots_Turoman.py            # Figures and console statistics
├── subject_npy/                # Output of conversion (input for LDA + NeuralNetwork)
├── EEGtrial_cleanEEGMAT_*.csv  # Raw CSV files (place here before converting)
└── results/
    ├── turoman_results.pkl
    ├── Figure2_TimeAveragedAccuracy.pdf
    ├── Figure5_ConfusionMatrices.pdf
    └── Figure6_IndividualAccuracy.pdf
```

---

## Step 1: Convert CSV to subject `.npy` files

`Convert_to_subject_cpy.py` reads the three condition CSV files and writes one NumPy file per subject per condition.

### Input CSV files

Place these files in the `TuromanClassifier/` directory (same folder as the script):

| Condition | Filename |
|---|---|
| BSL | `EEGtrial_cleanEEGMAT_BSL.csv` |
| SENSORY | `EEGtrial_cleanEEGMAT_SENSORY.csv` |
| DELAY | `EEGtrial_cleanEEGMAT_DELAY.csv` |

Each CSV row is one electrode × trial recording. Columns (last four metadata columns):

| Column | Content |
|---|---|
| `:-4` | EEG time samples |
| `-4` | Class label (`1` = Visual, `3` = Spatial, `5` = Verbal) |
| `-3` | Channel identifier |
| `-2` | Subject identifier |
| `-1` | Trial identifier |

### What the script does

1. Scans each CSV in chunks (`CHUNK_SIZE = 100_000`) to find all subject IDs
2. For each subject, filters rows and **reconstructs** a 3D tensor:
   - `X` shape: `(trials, channels, time)`
   - `y` shape: `(trials,)` with labels in `{1, 3, 5}`
3. Saves to `subject_npy/{CONDITION}_subject_{id}.npy` as:

```python
{"X": np.ndarray, "y": np.ndarray}
```

### Run conversion

```bash
cd TuromanClassifier
python Convert_to_subject_cpy.py
```

Example output file: `subject_npy/SENSORY_subject_1.npy`

### Use with NeuralNetwork

Copy (or symlink) all files from `subject_npy/` into `NeuralNetwork/data/`. Files must start with the condition prefix (e.g. `SENSORY_`) — the default naming from the converter satisfies this.

---

## Step 2: Input data format (after conversion)

Each `.npy` file contains:

```python
{
    "X": np.ndarray,  # shape (trials, channels, time)
    "y": np.ndarray,  # labels in {1, 3, 5}
}
```

---

## Step 3: Run LDA baseline

```bash
# Run LDA cross-validation (reads subject_npy/, saves results/turoman_results.pkl)
python TuromanClassifier.py

# Generate figures
python plots_Turoman.py
```

## Requirements

- NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn

---

