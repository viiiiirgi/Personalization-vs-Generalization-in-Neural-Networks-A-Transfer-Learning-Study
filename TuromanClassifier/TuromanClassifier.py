import numpy as np
import os
import time
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import ttest_1samp
#from sklearn.utils import resample

OUTPUT_DIR = "results"
DATA_DIR = "subject_npy"

CONDITIONS = ["BSL", "SENSORY", "DELAY"]
CLASSES = [1, 3, 5]  # Visual, Spatial, Verbal

N_ITERATIONS = 100 #for the repeated cross-validation
TRIALS_PER_AVG = 5 #pseudo-trial is the average of 5 real trials
TRIALS_COUNT = 100 # generate 100 pseudo-trials per class

CHANCE = 1/3

#np.random.seed(1)  remove for random resampling across iterations, not one fixed configuration (seed fixes pseudo-trial creation and fixes shuffle behavior)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#Match the paper's time windows (Baseline 0-300 ms, Sensory 300-1300 ms, Delay 1300-3300 ms)



# EEG is noisy -> averaging trials improves signal to noise ratio
def create_pseudo_trials(X, y, trials_per_avg=TRIALS_PER_AVG, n_trials= TRIALS_COUNT):

    X_new = []
    y_new = []

    classes = np.unique(y)

    for label in classes:

        idx = np.where(y == label)[0]

        if len(idx) < trials_per_avg:
            continue

        # shuffle once
        #np.random.shuffle(idx)

        # split into non-overlapping groups
        #n_groups = len(idx) // trials_per_avg

        #for i in range(n_groups):
        #    group = idx[i*trials_per_avg:(i+1)*trials_per_avg]
        #    X_new.append(X[group].mean(axis=0))
        #    y_new.append(label)

        # Create exactly n_trials pseudo-trials
        for _ in range(n_trials):

            sampled_idx = np.random.choice(
                idx,
                size=trials_per_avg,
                replace=False
            )

            pseudo = X[sampled_idx].mean(axis=0)

            X_new.append(pseudo)
            y_new.append(label)
        

    return np.array(X_new), np.array(y_new)

#def balance_classes(X, y):
#    X_bal, y_bal = [], []
#    max_n = max([np.sum(y == c) for c in np.unique(y)])

#    for c in np.unique(y):
#        X_c = X[y == c]
#        y_c = y[y == c]

#        X_up, y_up = resample(X_c, y_c,
#                             replace=True,
#                             n_samples=max_n,
#                             random_state=None)

#        X_bal.append(X_up)
#        y_bal.append(y_up)

#    return np.vstack(X_bal), np.hstack(y_bal)

# time-averaged decoding for one subject and one condition
def run_subject(npy_file, condition):

    #load subject data
    data = np.load(npy_file, allow_pickle=True).item()

    X = data["X"]   # (trials, channels, time)
    y = data["y"]

    n_trials, n_channels, n_times = X.shape
    print(X.shape)

    # Extract correct time window
    #start_t, end_t = get_time_window_indices(condition, n_times)
    
    # Average across time window
    #X_window = X[:, :, start_t:end_t].mean(axis=2)   # trials × channels

    # Average entire epoch
    X_window = X.mean(axis=2)

    if X_window.shape[1] == 0:
        raise ValueError("Time window selection is empty")

    # 100 independent random splits, 67% train and 33% test
    sss = StratifiedShuffleSplit(
        n_splits=N_ITERATIONS,
        test_size=0.33
    )

    iteration_scores = []
    conf_matrix_total = np.zeros((3, 3))

    for train_idx, test_idx in sss.split(X_window, y):


        X_train, X_test = X_window[train_idx], X_window[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create pseudo-trials separately for train and test
        X_train_avg, y_train_avg = create_pseudo_trials(X_train, y_train)
        X_test_avg, y_test_avg = create_pseudo_trials(X_test, y_test)

        #without pseudo trials
        #X_train_avg, y_train_avg = X_train.copy(), y_train.copy()
        #X_test_avg,  y_test_avg  = X_test.copy(),  y_test.copy()

        #balancing
        #X_train_avg, y_train_avg =balance_classes(X_train_avg, y_train_avg)
        #X_test_avg, y_test_avg =balance_classes(X_test_avg, y_test_avg)

        if len(np.unique(y_train_avg)) < 2:
            continue

        # Demean across trials (subtract the mean of training data from train and test to remove the amplitude differences and center features)
        train_mean = X_train_avg.mean(axis=0)
        X_train_avg -= train_mean
        X_test_avg -= train_mean

        # Train LDA
        lda = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto"
        )

        lda.fit(X_train_avg, y_train_avg)

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

    

    # average accuracy over 100 splits
    mean_acc = np.mean(iteration_scores)
    # Convert counts to proportions
    row_sums = conf_matrix_total.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_matrix_prop = conf_matrix_total / row_sums


    return mean_acc, conf_matrix_prop

def compute_statistics(subject_accuracies):

    subject_accuracies = np.array(subject_accuracies)

    mean_acc = np.mean(subject_accuracies)
    median_acc = np.median(subject_accuracies)
    sd_acc = np.std(subject_accuracies, ddof=1)

    t_val, p_val = ttest_1samp(subject_accuracies, CHANCE, alternative="greater")

    return {
        "mean": mean_acc,
        "median": median_acc,
        "sd": sd_acc,
        "t_value": t_val,
        "p_value": p_val
    }

# loops over all subjects in one condition
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

    print("\n--- Statistics vs chance ---")

    for condition in all_results:

        subject_accuracies = [res["accuracy"] for res in all_results[condition].values()]

        stats = compute_statistics(subject_accuracies)

        print(
            f"{condition}: "
            f"Mean accuracy: {stats['mean']*100:.2f}%"
            f" Median accuracy: {stats['median']*100:.2f}%"
            f" SD: {stats['sd']*100:.2f}%"
            f" t-value: {stats['t_value']:.2f}"
            f" p-value: {stats['p_value']:.4e}"
        )

        all_results[condition]["group_stats"] = stats

    with open(os.path.join(OUTPUT_DIR, "turoman_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    print(f"\nTotal time: {(time.time()-start)/60:.2f} minutes")
