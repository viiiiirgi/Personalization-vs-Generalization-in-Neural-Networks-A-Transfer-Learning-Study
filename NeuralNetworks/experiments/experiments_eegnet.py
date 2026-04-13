import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import ttest_1samp
from tensorflow.keras.optimizers import Adam
from utils.data_loader import load_subject
from utils.train_utils import (
    set_seed,
    normalize_train_test,
    create_pseudo_trials,
    build_model,
    clear
)

EPOCHS = 10
BATCH_SIZE = 8


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


def run_subject_dependent(file, model_name, n_runs=5):

    X, y = load_subject(file)

    chans = X.shape[1]
    samples = X.shape[2]

    accs = []
    cm_total = np.zeros((3,3))

    for seed in range(n_runs):
        set_seed(seed)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=seed
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed
            )

        X_train, X_test = normalize_train_test(X_train, X_test)

        X_train, y_train = create_pseudo_trials(X_train, y_train)
        X_test, y_test = create_pseudo_trials(X_test, y_test)

        model = build_model(model_name, chans, samples)

        model.fit(X_train, y_train, EPOCHS, BATCH_SIZE, verbose=0)

        preds = np.argmax(model.predict(X_test), axis=1)

        accs.append(accuracy_score(y_test, preds))
        cm_total += confusion_matrix(y_test, preds, labels=[0,1,2])

        clear()

    row_sums = cm_total.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_avg = cm_total / row_sums

    return np.mean(accs), cm_avg


def run_subject_independent(files, model_name, data_dir, n_runs=5):

    results = {}
    accuracies = []

    for test_file in files:

        X_test, y_test = load_subject(f"{data_dir}/{test_file}")

        X_train_list, y_train_list = [], []

        for f in files:
            if f == test_file:
                continue
            X_tmp, y_tmp = load_subject(f"{data_dir}/{f}")
            X_train_list.append(X_tmp)
            y_train_list.append(y_tmp)

        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)

        # shuffle
        perm = np.random.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        X_train_n, X_test_n = normalize_train_test(X_train, X_test)

        accs = []
        cm_total = np.zeros((3,3))

        for seed in range(n_runs):
            set_seed(seed)

            X_train_p, y_train_p = create_pseudo_trials(X_train_n, y_train)
            X_test_p, y_test_p = create_pseudo_trials(X_test_n, y_test)

            model = build_model(model_name, X_train.shape[1], X_train.shape[2])

            model.fit(X_train_p, y_train_p, EPOCHS, BATCH_SIZE, verbose=0)

            preds = np.argmax(model.predict(X_test_p), axis=1)

            acc = accuracy_score(y_test_p, preds)
            accs.append(acc)

            cm_total += confusion_matrix(y_test_p, preds, labels=[0,1,2])

            clear()

        row_sums = cm_total.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_avg = cm_total / row_sums

        mean_acc = np.mean(accs)
        accuracies.append(mean_acc)

        results[test_file] = {
            "accuracy": mean_acc,
            "confusion_matrix": cm_avg
        }

        print(f"{test_file} → {mean_acc:.4f}")

    results["group_stats"] = compute_statistics(accuracies)
    return results


def run_transfer_learning(files, model_name, data_dir, n_runs=5, fine_tune_split=0.3):

    results = {}
    accuracies = []

    for test_file in files:

        print(f"\n--- Transfer Learning on {test_file} ---")

        # Load target subject
        X_target, y_target = load_subject(f"{data_dir}/{test_file}")

        accs = []
        cm_total = np.zeros((3,3))

        for seed in range(n_runs):
            set_seed(seed)

            # Split target subject (fine-tune + test)
            try:
                X_ft, X_test, y_ft, y_test = train_test_split(
                    X_target, y_target,
                    test_size=fine_tune_split,
                    stratify=y_target,
                    random_state=seed
                )
            except ValueError:
                X_ft, X_test, y_ft, y_test = train_test_split(
                    X_target, y_target,
                    test_size=fine_tune_split,
                    random_state=seed
                )

            # Build pretraining dataset (all other subjects)
            X_train_list, y_train_list = [], []

            for f in files:
                if f == test_file:
                    continue
                X_tmp, y_tmp = load_subject(f"{data_dir}/{f}")
                X_train_list.append(X_tmp)
                y_train_list.append(y_tmp)

            X_pretrain = np.concatenate(X_train_list)
            y_pretrain = np.concatenate(y_train_list)

            # Normalize
            X_pretrain, X_test = normalize_train_test(X_pretrain, X_test)
            _, X_ft = normalize_train_test(X_pretrain, X_ft)

            # Pseudo trials
            X_pre_p, y_pre_p = create_pseudo_trials(X_pretrain, y_pretrain)
            X_ft_p, y_ft_p = create_pseudo_trials(X_ft, y_ft)
            X_test_p, y_test_p = create_pseudo_trials(X_test, y_test)

            # Build model
            model = build_model(model_name, X_pretrain.shape[1], X_pretrain.shape[2])

            #Pretraining
            model.fit(X_pre_p, y_pre_p, EPOCHS, BATCH_SIZE, verbose=0)

            #Fine-tuning 
            for layer in model.layers[:-2]: # Freeze early layers 
                layer.trainable = False

            model.compile(
                optimizer=Adam(1e-4),  # lower LR for fine-tuning
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            model.fit(X_ft_p, y_ft_p, EPOCHS, BATCH_SIZE, verbose=0)

            # Evaluation 
            preds = np.argmax(model.predict(X_test_p), axis=1)

            acc = accuracy_score(y_test_p, preds)
            accs.append(acc)

            cm_total += confusion_matrix(y_test_p, preds, labels=[0,1,2])

            clear()

        # Normalize confusion matrix
        row_sums = cm_total.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_avg = cm_total / row_sums

        mean_acc = np.mean(accs)
        accuracies.append(mean_acc)

        results[test_file] = {
            "accuracy": mean_acc,
            "confusion_matrix": cm_avg
        }

        print(f"{test_file} → {mean_acc:.4f}")

    results["group_stats"] = compute_statistics(accuracies)
    return results