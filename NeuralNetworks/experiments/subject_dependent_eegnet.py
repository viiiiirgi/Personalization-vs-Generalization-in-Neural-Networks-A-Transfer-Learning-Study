import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from scipy.stats import ttest_1samp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.eegnet import EEGNet
from utils.data_loader import load_subject

import random

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


DATA_DIR = "../data"
RESULTS_DIR="../results"
RESULTS = {}

CONDITION = "BSL"  # set to "BSL" or "DELAY" or "SENSORY"
EXPERIMENT_NAME = f"eegnet_{CONDITION}_sd"

EPOCHS = 10
BATCH_SIZE = 8


'''
def create_pseudo_trials(X, y, n_pseudo=100, trials_per_avg=5):
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
'''

def run_subject(file, n_runs=1):

    #takes one data file
    X, y = load_subject(file)

    #start_t, end_t = get_time_window_indices(CONDITION, X.shape[2])
    #X = X[:, :, start_t:end_t]

    

    accs = []
    chans = X.shape[1]
    samples = X.shape[2]
    cm_total = np.zeros((3,3))

    # stratified split can fail if a class has too few trials; fall back safely
    for seed in range(n_runs):
        set_seed(seed)
        try:
            # X_train, y_train = create_pseudo_trials(X_train, y_train)
            #X_test, y_test = create_pseudo_trials(X_test, y_test)

            # 70% training, 30% for testing
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                stratify=y,
                random_state=seed
            )

            mean = X_train.mean()
            std = X_train.std() + 1e-8

            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=seed
            )

        model = EEGNet(
            nb_classes=3,
            Chans=chans,
            Samples=samples
        )

        # Prepere eegnet: give it Adam optimizer and the loss function
        model.compile(
            optimizer=Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # actual study: the model looks at the training data
        model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.0,
            verbose=0
        )

        preds = model.predict(X_test) # give to the model the testing data to guess the memory condition for each trial
        preds = np.argmax(preds, axis=1)

        # Grade the results
        acc = accuracy_score(y_test, preds)
        accs.append(acc)
        cm = confusion_matrix(y_test, preds, labels=[0,1,2])
        cm_total +=cm

        # clean memory to help the RAM to not crash 
        K.clear_session()

        if seed % 10 == 0:
            print(f"  Run {seed}/{n_runs}")

    row_sums =cm_total.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1
    cm_avg = cm_total/row_sums
    return np.mean(accs), cm_avg


def compute_statistics(subject_accuracies):

    subject_accuracies = np.array(subject_accuracies)

    mean_acc = np.mean(subject_accuracies)
    median_acc = np.median(subject_accuracies)
    sd_acc = np.std(subject_accuracies, ddof=1)

    t_val, p_val = ttest_1samp(subject_accuracies, 1/3)

    return {
        "mean": mean_acc,
        "median": median_acc,
        "sd": sd_acc,
        "t_value": t_val,
        "p_value": p_val
    }


def process_file(f):
    file_path = os.path.join(DATA_DIR, f)

    print(f"\nProcessing {f}...")

    acc, cm = run_subject(file_path)

    return f, acc, cm


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".npy") and f.startswith(CONDITION + "_")
    )

    subject_results = {}
    accuracies = []

    #  Parallel execution across subjects
    results = []

    for f in files: results.append(process_file(f))

    # Collect results
    for f, acc, cm in results:

        subject_results[f] = {
            "accuracy": acc,
            "confusion_matrix": cm
        }

        accuracies.append(acc)

        print(f"{f} → Accuracy: {acc:.4f}")

    # Group statistics 
    stats = compute_statistics(accuracies)

    print("\n--- Group statistics ---")
    print(
        f"Mean: {stats['mean']:.4f} | "
        f"Median: {stats['median']:.4f} | "
        f"SD: {stats['sd']:.4f} | "
        f"t-value: {stats['t_value']:.2f} | "
        f"p-value: {stats['p_value']:.4e}"
    )

    subject_results["group_stats"] = stats

    # Save results
    save_path = os.path.join(RESULTS_DIR, EXPERIMENT_NAME + ".npy")
    np.save(save_path, subject_results)

    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()