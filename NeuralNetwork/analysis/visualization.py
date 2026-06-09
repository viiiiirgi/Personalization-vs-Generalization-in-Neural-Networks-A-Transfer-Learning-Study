import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.train_utils import build_model

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SUBJECT = "SENSORY_subject_1383.npy"

EXPERIMENTS = [
    "AllData_NoPseudo",
    "AllData_PseudoTrainOnly",
    "AllData_PseudoTrainAndTest",
    "Downsample_1000_NoPseudo",
    "Downsample_MinSubject_NoPseudo",
]

STRATEGIES = ["sd", "si", "tl"]


# DATA
def load_subject():
    data = np.load(os.path.join("data", SUBJECT),allow_pickle=True).item()

    X = data["X"].astype(np.float32)
    y = np.vectorize({1: 0, 3: 1, 5: 2}.get)(data["y"]).astype(np.int64)

    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-6
    X_norm = (X - mean) / std
    X_norm = X_norm[..., np.newaxis]

    return X_norm, y, X.shape[1], X.shape[2]


# MODEL
def load_model(model_path, chans, samples, dropout):
    model = build_model("eegnet", chans, samples, dropout=dropout)
    model.to(DEVICE)
    dummy = torch.zeros(1, chans, samples, 1, device=DEVICE)

    with torch.no_grad():
        model(dummy)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    return model


# FEATURE VISUALIZATION
def plot_pca(model, X, y, save_file, title):
    X_t = torch.tensor( X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        feats = model.extract_features(X_t)

    feats = feats.cpu().numpy()
    pca = PCA(n_components=2)
    proj = pca.fit_transform(feats)

    plt.figure(figsize=(7, 6))

    for cls in np.unique(y):
        idx = y == cls
        plt.scatter(proj[idx, 0], proj[idx, 1], label=f"Class {cls}", alpha=0.7)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()


# TEMPORAL FILTERS
def plot_temporal_filters(model, save_file, sfreq=256):

    weights = model.conv1.weight.detach().cpu().numpy()
    n_filters = weights.shape[0]
    fig, axes = plt.subplots( n_filters, 2, figsize=(10, 3*n_filters))

    if n_filters == 1:
        axes = np.array([axes])

    for i in range(n_filters):
        kernel = weights[i, 0, 0, :]

        axes[i, 0].plot(kernel)
        axes[i, 0].set_title(f"Filter {i+1}")

        fft = np.abs(np.fft.rfft(kernel))
        freqs = np.fft.rfftfreq(len(kernel), d=1/sfreq)

        axes[i, 1].plot(freqs, fft)
        axes[i, 1].set_xlim(0, 60)
        axes[i, 1].set_title(f"Frequency {i+1}")

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()


# CHANNEL IMPORTANCE
def plot_channel_importance(model, save_file):
    weights = (model.depthwise.weight.detach().cpu().numpy())
    weights = weights.squeeze()

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(weights), aspect="auto")
    plt.colorbar()
    plt.xlabel("Channel")
    plt.ylabel("Spatial Filter")
    plt.title("Channel Importance")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()


# SALIENCY MAP
def plot_saliency_map(model, X, save_file):
    X_t = torch.tensor(X[:50], dtype=torch.float32, device=DEVICE, requires_grad=True)

    output = model(X_t)
    pred = output.argmax(dim=1)
    score = output.gather(1, pred.unsqueeze(1)).sum()
    score.backward()
    saliency = (X_t.grad.abs().mean(0).squeeze(-1).cpu().numpy())
    
    plt.figure(figsize=(12, 6))
    plt.imshow( saliency, aspect="auto", origin="lower")
    plt.colorbar()
    plt.xlabel("Time Samples")
    plt.ylabel("Channels")
    plt.title("Saliency Map")
    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()


# MAIN
def main():
    X, y, chans, samples = load_subject()

    feature_dir = os.path.join("analysis", "feature_plots")
    temporal_dir = os.path.join("analysis", "temporal_filters")
    channel_dir = os.path.join("analysis","channel_importance")
    saliency_dir = os.path.join("analysis", "saliency_maps")

    for d in [feature_dir, temporal_dir, channel_dir, saliency_dir]:
        os.makedirs(d, exist_ok=True)

    for experiment in EXPERIMENTS:
        for strategy in STRATEGIES:
            dropout = (0.5 if strategy == "sd" else 0.25)
            model_path = os.path.join("saved_models", experiment, f"{strategy}_{SUBJECT}_best.pt")

            if not os.path.exists(model_path):
                print("Missing:", model_path)
                continue

            print("Loading:", model_path)

            model = load_model(model_path, chans, samples, dropout)
            base_name = (f"{experiment}_{strategy}2.png")

            plot_pca(model, X, y, os.path.join(feature_dir,base_name),f"{experiment} - {strategy.upper()}")
            plot_temporal_filters( model, os.path.join(temporal_dir, base_name))
            plot_channel_importance(model, os.path.join(channel_dir, base_name))
            plot_saliency_map(model, X, os.path.join(saliency_dir, base_name))

if __name__ == "__main__":
    main()