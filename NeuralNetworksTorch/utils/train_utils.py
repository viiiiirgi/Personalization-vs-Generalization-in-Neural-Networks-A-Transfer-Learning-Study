import numpy as np
import random
import torch
from scipy.stats import ttest_1samp
from models.eegnet_torch import EEGNet

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=(0,2), keepdims=True) + 1e-6

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


def create_pseudo_trials(X, y, n_pseudo=1, trials_per_avg=5):
    X_new, y_new = [], []
    classes = np.unique(y)

    for label in classes:
        idx = np.where(y == label)[0]
        if len(idx) == 0:
            continue

        for _ in range(n_pseudo):
            sampled = np.random.choice(idx, size=trials_per_avg, replace=True)
            X_new.append(X[sampled].mean(axis=0))
            y_new.append(label)

    return np.array(X_new), np.array(y_new)


def build_model(model_name, chans, samples):
    if model_name == "eegnet":
        model = EEGNet(nb_classes=3, Chans=chans, Samples=samples)
    elif model_name == "tcn":
        from models.tcn import TCN
        model = TCN(nb_classes=3, Chans=chans, Samples=samples)
    elif model_name == "cfc":
        from models.cfc import CfC
        model = CfC(nb_classes=3, Chans=chans, Samples=samples)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model

def predict_model(model, X):
    device = next(model.parameters()).device
    X = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)

    return preds.cpu().numpy()

def train_model(model, X, y, epochs, batch_size, patience=5, val_split=0.2,lr=1e-3):
    device = next(model.parameters()).device

    # split train / validation
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - val_split))

    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y[train_idx], dtype=torch.long).to(device)

    X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y[val_idx], dtype=torch.long).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    best_model_state = None

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        perm = torch.randperm(X_train.size(0))

        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]

            xb = X_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

        # ---- VALIDATION ----
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            val_loss = criterion(outputs, y_val).item()

        # ---- EARLY STOPPING ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            # restore best model
            model.load_state_dict(best_model_state)
            break

def compute_statistics(subject_accuracies):
    subject_accuracies = np.array(subject_accuracies)

    mean_acc = np.mean(subject_accuracies)
    median_acc = np.median(subject_accuracies)
    sd_acc = np.std(subject_accuracies, ddof=1)

    t_val, p_val = ttest_1samp(subject_accuracies, 1/3) #tests if accuracy is above chance

    return {
        "mean": mean_acc,
        "median": median_acc,
        "sd": sd_acc,
        "t_value": t_val,
        "p_value": p_val
    }