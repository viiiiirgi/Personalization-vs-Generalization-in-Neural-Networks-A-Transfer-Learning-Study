import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from models.eegnet import EEGNet
# for gpu
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras.callbacks import EarlyStopping

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def normalize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=(0,2), keepdims=True) + 1e-6

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


def create_pseudo_trials(X, y, n_pseudo=30, trials_per_avg=5):
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

    model.compile(
        optimizer=Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def clear():
    K.clear_session()