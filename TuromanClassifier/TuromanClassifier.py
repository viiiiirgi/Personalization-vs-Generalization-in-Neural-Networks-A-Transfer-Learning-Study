import numpy as np
import os
import time
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix


DATA_DIR = "subject_npy"
CONDITIONS = ["BSL", "SENSORY", "DELAY"]
CLASSES = [1, 3, 5]  # Visual, Spatial, Verbal

N_ITERATIONS = 100
TRIALS_PER_AVG = 5
TRIALS_COUNT = 100

np.random.seed(42)


def get_time_window_indices(condition, n_times):

    sampling_rate = n_times / 3.3

    if condition == "BSL":
        return 0, int(0.3 * sampling_rate)

    elif condition == "SENSORY":
        return int(0.3 * sampling_rate), int(1.3 * sampling_rate)

    elif condition == "DELAY":
        return int(1.3 * sampling_rate), int(3.3 * sampling_rate)



def create_pseudo_trials(X, y, n_pseudo=100, trials_per_avg=5):

    X_new = []
    y_new = []

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



def run_subject(npy_file, condition):

    data = np.load(npy_file, allow_pickle=True).item()

    X = data["X"]   # (trials, channels, time)
    y = data["y"]

    n_trials, n_channels, n_times = X.shape

    # Extract correct time window
    start_t, end_t = get_time_window_indices(condition, n_times)
    
    # Average across time window
    X_window = X[:, :, start_t:end_t].mean(axis=2)   # trials × channels

    if X_window.shape[1] == 0:
        raise ValueError("Time window selection is empty")


    sss = StratifiedShuffleSplit(
        n_splits=N_ITERATIONS,
        test_size=0.33
    )

    iteration_scores = []
    conf_matrix_total = np.zeros((3, 3))


    for train_idx, test_idx in sss.split(X_window, y):

        X_train, X_test = X_window[train_idx], X_window[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create pseudo-trials (5 per average)
        X_train_avg, y_train_avg = create_pseudo_trials(X_train, y_train)
        X_test_avg, y_test_avg = create_pseudo_trials(X_test, y_test)

        if len(np.unique(y_train_avg)) < 2:
            continue

        # Demean across trials (train mean only)
        train_mean = X_train_avg.mean(axis=0)
        X_train_avg -= train_mean
        X_test_avg -= train_mean

        # Oversample minority class (training only)
        ros = RandomOverSampler()
        X_train_bal, y_train_bal = ros.fit_resample(
            X_train_avg,
            y_train_avg
        )

        # Train LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_bal, y_train_bal)

        preds = lda.predict(X_test_avg)

        iteration_scores.append(
            accuracy_score(y_test_avg, preds)
        )

        cm = confusion_matrix(
            y_test_avg,
            preds,
            labels=CLASSES
        )

        conf_matrix_total += cm

    mean_acc = np.mean(iteration_scores)
    # Convert counts to proportions
    row_sums = conf_matrix_total.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_matrix_prop = conf_matrix_total / row_sums


    return mean_acc, conf_matrix_prop



def run_condition(condition):

    print(f"\n========== {condition} ==========")

    files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.startswith(condition)
    ])

    results = {}

    for file in files:

        subject_id = file.split("_")[-1].replace(".npy", "")
        print(f"Subject {subject_id}")

        full_path = os.path.join(DATA_DIR, file)

        acc, cm = run_subject(full_path, condition)

        results[subject_id] = {
            "accuracy": acc,
            "confusion": cm
        }

    return results



if __name__ == "__main__":

    start = time.time()

    all_results = {}

    for condition in CONDITIONS:
        all_results[condition] = run_condition(condition)

    print("\n--- Final Mean Accuracies ---")

    for condition in all_results:

        subject_accuracies = [res["accuracy"] for res in all_results[condition].values()]

        print(
            f"{condition}: "
            f"{np.mean(subject_accuracies)*100:.2f}%"
        )

    with open("turoman_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print(f"\nTotal time: {(time.time()-start)/60:.2f} minutes")
