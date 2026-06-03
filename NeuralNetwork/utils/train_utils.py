import numpy as np
import random
import torch
from scipy.stats import ttest_1samp
from models.eegnet_torch import EEGNet
from sklearn.model_selection import train_test_split

# fixes randomness acrosss libraries
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_train_test(X_train, X_test): 
    #normalization per channel (just 0 for per channel per time)
    mean = X_train.mean(axis=(0,2), keepdims=True)
    std = X_train.std(axis=(0,2), keepdims=True) + 1e-6

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test

def create_pseudo_trials(X, y, trials_per_avg=5, n_trials=100, seed=None):

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    X_new, y_new = [], []
    classes = np.unique(y)

    for label in classes:
        idx = np.where(y == label)[0]

        #split into groups of size 5
        if len(idx) < trials_per_avg:
            continue

        for _ in range(n_trials):
            # Randomly select trials for the average
            # replace=False ensures the same trial isn't picked twice for the SAME pseudo-trial
            sampled_idx = rng.choice(idx, size=trials_per_avg, replace=False)
            
            # Average the selected trials across the trial dimension
            pseudo = X[sampled_idx].mean(axis=0)
            
            X_new.append(pseudo)
            y_new.append(label)

    return np.array(X_new), np.array(y_new)

def subsample_data(X, y, target_size, seed):
    np.random.seed(seed)
    idx = np.random.choice(len(X), target_size, replace=False)
    return X[idx], y[idx]

def compute_sd_train_size(files, all_data, seed=0):
    sizes = []

    for f in files:
        data = all_data[f]
        X = data["X"]
        y = np.vectorize({1:0,3:1,5:2}.get)(data["y"])

        try:
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=seed
            )
        except ValueError:
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.3, random_state=seed
            )

        X_tr, _, _, _ = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed
        )

        sizes.append(len(X_tr))

    return min(sizes) 

def build_model(model_name, chans, samples, dropout=0.5):
    if model_name == "eegnet":
        model = EEGNet(nb_classes=3, Chans=chans, Samples=samples, dropoutRate=dropout)
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

def train_model(model, X, y, epochs, batch_size,
                X_val=None, y_val=None,
                lr=2e-3, optimizer=None, patience=3):

    device = next(model.parameters()).device

    torch.backends.cudnn.benchmark = True

    X_train = torch.tensor(X, dtype=torch.float32).to(device)
    y_train = torch.tensor(y, dtype=torch.long).to(device)

    use_validation = X_val is not None and y_val is not None

    if use_validation:
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):

        # TRAIN
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

        # VALIDATION (only if provided)
        if use_validation:
            model.eval()
            with torch.no_grad():
                outputs = model(X_val)
                val_loss = criterion(outputs, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                model.load_state_dict(best_model_state)
                break

    return model

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