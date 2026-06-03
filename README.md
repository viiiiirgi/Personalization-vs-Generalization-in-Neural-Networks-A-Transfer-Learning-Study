# Personalization vs Generalization in Neural Networks — Transfer Learning Study

Thesis project comparing **subject-dependent**, **subject-independent**, and **transfer-learning** EEG decoding with EEGNet, against an LDA baseline (Turoman et al., 2024).

## Repository layout

```text
Personalization-vs-Generalization-in-Neural-Networks-A-Transfer-Learning-Study/
├── TuromanClassifier/          # Step 1–2: CSV → .npy conversion + LDA baseline
│   ├── Convert_to_subject_cpy.py
│   ├── TuromanClassifier.py
│   └── plots_Turoman.py
└── NeuralNetwork/              # Step 3: EEGNet experiments (SD / SI / TL)
    ├── utils/data_loader.py
    ├── utils/run_experiments.py
    └── experiments/experiments_eegnet.py
```

## End-to-end workflow

```text
Raw CSV files (BSL, SENSORY, DELAY)
        │
        ▼  TuromanClassifier/Convert_to_subject_cpy.py
subject_npy/*.npy  (one file per subject × condition)
        │
        ├──► TuromanClassifier/TuromanClassifier.py  → LDA baseline results
        │
        └──► copy into NeuralNetwork/data/
                    │
                    ▼  NeuralNetwork/utils/run_experiments.py
             results/eegnet/<experiment>/  (accuracies, figures, tables, visualizations)
```

1. **Convert CSV → subject `.npy` files** — see [TuromanClassifier/README.md](TuromanClassifier/README.md)
2. **Run LDA baseline**  — `TuromanClassifier/TuromanClassifier.py`
3. **Copy `.npy` files** into `NeuralNetwork/data/`
4. **Run EEGNet experiments** — see [NeuralNetwork/README.MD](NeuralNetwork/README.MD)

## Requirements

- Python 3.8+
- PyTorch, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn

See each subfolder README for detailed usage.
