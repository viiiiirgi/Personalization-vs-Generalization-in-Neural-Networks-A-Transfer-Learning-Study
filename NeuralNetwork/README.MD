# NetworksTorch — EEGNet Decoding (SD / SI / Transfer Learning)

PyTorch implementation of **EEGNet** for comparing three training strategies on the same EEG memory-decoding dataset used in the Turoman LDA baseline.

## Project structure

```text
NeuralNetwork/
├── data/                          # {CONDITION}_*.npy (copy from TuromanClassifier/subject_npy/)
├── models/
│   └── eegnet_torch.py            # EEGNet + forward_with_activations()
├── experiments/
│   └── experiments_eegnet.py      # SD, SI, TL runners + EXPERIMENTS config
├── utils/
│   ├── data_loader.py             # Load .npy + remap labels + add CNN channel dim
│   ├── train_utils.py             # Training, pseudo-trials, normalization
│   ├── run_experiments.py         # Main entry point
│   ├── plot_like_turoman.py       # Figures + LaTeX tables from result .npy
│   ├── plot_finetune_curv.py      # Fine-tuning fraction curves
│   ├── feature_viz.py             # Filter / activation / embedding plots
│   └── visualize_features.py      # Optional standalone viz CLI
└── results/eegnet/
    └── <experiment_name>/
        ├── BSL_sd.npy, BSL_si.npy, BSL_tl.npy
        ├── sd_Figure2_Accuracy.pdf, …
        ├── sd_MainStats.tex, …
        └── visualizations/<CONDITION>/   # if GENERATE_VIZ=True
```

---

## Step 1: Prepare data

Run `TuromanClassifier/Convert_to_subject_cpy.py` first, then copy files:

```bash
cp TuromanClassifier/subject_npy/*.npy NeuralNetwork/data/
```

Each file must be named `{CONDITION}_*.npy` (e.g. `SENSORY_subject_1.npy`) and contain:

```python
{"X": (trials, channels, time), "y": labels in {1, 3, 5}}
```

---

## Step 2: Data loader (`utils/data_loader.py`)

`load_subject(file)` is the standard way to load one subject file for EEGNet:

```python
from utils.data_loader import load_subject

X, y = load_subject("data/SENSORY_subject_1.npy")
# X: (trials, channels, time, 1)  float32  — CNN input shape
# y: (trials,)                    int64    — labels in {0, 1, 2}
```

What it does:

| Step | Detail |
|---|---|
| Load | Reads `.npy` dict with keys `X`, `y` |
| Labels | Maps `{1→0, 3→1, 5→2}` (Visual / Spatial / Verbal) |
| Dtype | `X` → `float32`, `y` → `int64` |
| CNN dim | Appends channel axis: `X[..., np.newaxis]` |

The experiment scripts (`experiments_eegnet.py`) apply the same mapping inline when preloading all subjects into RAM. Use `load_subject()` when loading a single file in a notebook or custom script.

---

## Experimental configurations

Defined in `experiments/experiments_eegnet.py` → `EXPERIMENTS`:

| Key | Pseudo train | Pseudo test | Pretrain downsampling |
|---|---|---|---|
| `AllData_NoPseudo` | No | No | None (full data) |
| `AllData_PseudoTrainOnly` | Yes | No | None |
| `AllData_PseudoTrainAndTest` | Yes | Yes | None |
| `Downsample_1000_NoPseudo` | No | No | Fixed 1000 trials |
| `Downsample_MinSubject_NoPseudo` | No | No | Matched to min SD train size |

Fine-tuning sweep (`10%–100%` of fine-tune pool) runs only for `AllData_NoPseudo` → saved under `FT_7/`, …, `FT_70/`.

---

## Training strategies

### Subject-dependent (SD)
- 70/30 stratified train/test per subject
- 20% of train → validation
- Dropout **0.5**, 10 runs (seeds 0–9)

### Subject-independent (SI)
- Leave-one-subject-out
- 20% validation from training subjects
- Dropout **0.25**, optional pretrain downsampling

### Transfer learning (TL)
- Pretrain on all other subjects (20 epochs max, lr `2×10⁻³`)
- Fine-tune **FC layer only** (conv frozen, lr `5×10⁻⁴`, 30 epochs max)
- Target: 70% fine-tune / 30% test
- Dropout **0.25**

Common: Adam, batch size 64, cross-entropy, early stopping (patience 3).

---

## Usage

Run from the `NeuralNetwork/` directory:

```bash
cd NeuralNetwork

# Run all conditions × all experiments (SD + SI + TL)
python -m utils.run_experiments

# Generate figures and LaTeX tables
python utils/plot_like_turoman.py

# Fine-tuning curve (edit paths at top of file first)
python utils/plot_finetune_curv.py
```

### Feature visualization

Integrated (default) — set in `utils/run_experiments.py`:

```python
GENERATE_VIZ = True   # False = identical to pre-viz behaviour, no extra output
```

Output folder:

```text
results/eegnet/<experiment>/visualizations/<CONDITION>/
  <subject>_sd_temporal_filters.png
  filter_similarity.png
  embedding_pca.png
  <subject>_sd_state_dict.pt
  <subject>_sd_eval.npz
```

Standalone (optional):

```bash
python -m utils.visualize_features --condition SENSORY --experiment AllData_NoPseudo
python -m utils.visualize_features --condition SENSORY --experiment AllData_NoPseudo --replot-only
```

## Requirements

- PyTorch
- NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn
