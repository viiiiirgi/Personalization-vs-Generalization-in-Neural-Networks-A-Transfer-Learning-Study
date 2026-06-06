import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from utils.train_utils import build_model
from models.eegnet_torch import EEGNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUBJECT = "SENSORY_subject_1383.npy"

EXPERIMENTS = [
    "AllData_NoPseudo",
    "AllData_PseudoTrainOnly",
    "AllData_PseudoTrainAndTest",
    "Downsample_1000_NoPseudo",
    "Downsample_MinSubject_NoPseudo",
]

STRATEGIES = ["sd", "si", "tl"]

def load_subject():
    data = np.load(
        os.path.join("data", SUBJECT),
        allow_pickle=True
    ).item()

    X = data["X"].astype(np.float32)

    y = np.vectorize({1: 0, 3: 1, 5: 2}.get)(
        data["y"]
    ).astype(np.int64)

    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-6

    X = (X - mean) / std
    X = X[..., np.newaxis]

    return X, y

def load_model(model_path, chans, samples, drop):

    model = build_model(
        "eegnet",
        chans,
        samples,
        dropout=drop
    )

    dummy = torch.zeros(
        1,
        chans,
        samples,
        1,
        device=DEVICE
    )

    with torch.no_grad():
        model(dummy)

    model.load_state_dict(
        torch.load(model_path,map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return model


def get_features(model, X):

    X_t = torch.tensor(X,dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        feats = model.extract_features(X_t)

    return feats.cpu().numpy()


def plot_pca(features, labels, title, save_file):

    pca = PCA(n_components=2)

    proj = pca.fit_transform(features)

    plt.figure(figsize=(7,6))

    for cls in np.unique(labels):

        idx = labels == cls

        plt.scatter(
            proj[idx,0],
            proj[idx,1],
            alpha=0.7,
            label=f"Class {cls}"
        )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()


def main():
    X, y = load_subject()

    os.makedirs("feature_plots", exist_ok=True)

    for experiment in EXPERIMENTS:

        for strategy in STRATEGIES:
            if strategy=="sd":
                dropout=0.5
            else :
                dropout=0.25

            model_path = os.path.join(
                "saved_models",
                experiment,
                f"{strategy}_{SUBJECT}_best.pt"
            )

            if not os.path.exists(model_path):
                print("Missing:", model_path)
                continue

            print("Loading:", model_path)

            model = load_model(
                model_path,
                X.shape[1],
                X.shape[2],
                dropout
            )

            features = get_features(model, X)

            plot_pca(
                features,
                y,
                f"{experiment} - {strategy.upper()}",
                os.path.join(
                    "feature_plots",
                    f"{experiment}_{strategy}.png"
                )
            )


if __name__ == "__main__":
    main()