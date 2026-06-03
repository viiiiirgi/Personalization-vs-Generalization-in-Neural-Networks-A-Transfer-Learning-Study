import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.train_utils import (
    set_seed,
    normalize_train_test,
    create_pseudo_trials,
    build_model,
    train_model,
    predict_model,
    compute_statistics,
    subsample_data
)
import gc
import os
import torch

EPOCHS_PRETRAIN = 20
EPOCHS = 30
BATCH_SIZE = 64

EXPERIMENTS = {

    "AllData_NoPseudo": {
        "use_pseudo_train": False,
        "use_pseudo_test": False,
        "downsample_mode": None,
        "fixed_trials": None,
    },

    "AllData_PseudoTrainAndTest": {
        "use_pseudo_train": True,
        "use_pseudo_test": True,
        "downsample_mode": None,
        "fixed_trials": None,
    },

    "AllData_PseudoTrainOnly": {
        "use_pseudo_train": True,
        "use_pseudo_test": False,
        "downsample_mode": None,
        "fixed_trials": None,
    },

    "Downsample_1000_NoPseudo": {
        "use_pseudo_train": False,
        "use_pseudo_test": False,
        "downsample_mode": "fixed",
        "fixed_trials": 1000,
    },

    "Downsample_MinSubject_NoPseudo": {
        "use_pseudo_train": False,
        "use_pseudo_test": False,
        "downsample_mode": "match_sd",
        "fixed_trials": None,
    }
}

FINE_TUNE_PERCENTS = [0.1, 0.25, 0.5, 0.75, 1.0]


# train and test on the same subject
def run_subject_dependent(file, model_name, config, n_runs=10, all_data=None):

    use_pseudo_train = config["use_pseudo_train"]
    use_pseudo_test = config["use_pseudo_test"]

    if all_data is not None:
        # extract filename only (because keys are filenames, not full paths)
        key = os.path.basename(file)
        data = all_data[key]
    else:
        data = np.load(file, allow_pickle=True).item()

    X = data["X"].astype(np.float32)
    y = np.vectorize({1:0,3:1,5:2}.get)(data["y"]).astype(np.int64)
    X = X[..., np.newaxis] #X=EEG data (trials x channel x time), y=labeles

    accs = [] #accuracy per run
    cm_total = np.zeros((3,3)) #confusion matrices

    for seed in range(n_runs):
        set_seed(seed)

        # random splits
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=seed
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed
            )

        # normalization
        X_train_n, X_test_n = normalize_train_test(X_train, X_test)
        
        # split TRAIN into train/val BEFORE pseudo-trials
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_n, y_train, test_size=0.2, stratify=y_train, random_state=seed
        )

        if use_pseudo_train:
            X_tr_p, y_tr_p = create_pseudo_trials(X_tr, y_tr, seed=seed)
            X_val_p, y_val_p = create_pseudo_trials(X_val, y_val, seed=seed+1)
        else:
            #no pseudo trials on train
            X_tr_p, y_tr_p = X_tr, y_tr
            X_val_p, y_val_p = X_val, y_val

        if use_pseudo_test:
            X_test_p, y_test_p = create_pseudo_trials(X_test_n, y_test, seed=seed+2)
        else:
            #no pseudo trials on test
            X_test_p, y_test_p = X_test_n, y_test   

        # creates the eegnet, the TCN or the CfC
        model = build_model(model_name, X_tr_p.shape[1], X_tr_p.shape[2], dropout=0.5)

        #Early stopping: stops if validation loss doesn't improve
        train_model(model, X_tr_p, y_tr_p, EPOCHS, BATCH_SIZE, X_val=X_val_p, y_val=y_val_p)

        # convert probabilities into class labeles
        preds = predict_model(model, X_test_p)

        #store results across runs
        accs.append(accuracy_score(y_test_p, preds))
        cm_total += confusion_matrix(y_test_p, preds, labels=[0,1,2])

        del model
        torch.cuda.empty_cache()

    #final confusion matrix: normalizes per class and make percentages
    row_sums = cm_total.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_avg = cm_total / row_sums
    per_class_acc = np.diag(cm_avg)

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    sem = std_acc / np.sqrt(len(accs))
    ci95 = 1.96 * sem

    results = {
        "accuracy": mean_acc,
        "std_accuracy": std_acc,
        "ci95": ci95,
        "confusion_matrix": cm_avg,
        "per_class_accuracy": per_class_acc,
        "run_accuracies": accs,
        "n_runs": len(accs)
    }

    #return np.mean(accs), cm_avg
    return results

# train on all subjects except one, test on that one
def run_subject_independent(files, model_name, data_dir, config, n_runs=10, all_data=None, sd_train_size=None):

    use_pseudo_train = config["use_pseudo_train"]
    use_pseudo_test = config["use_pseudo_test"]
    downsample_mode = config["downsample_mode"]
    fixed_trials = config["fixed_trials"]

    results = {}
    accuracies = []

    for test_file in files: # leave one subject out

        # select test subject
        data = all_data[test_file]
        X = data["X"].astype(np.float32)
        y = np.vectorize({1:0,3:1,5:2}.get)(data["y"]).astype(np.int64)
        X = X[..., np.newaxis]

        X_test, y_test = X, y

        X_train_list, y_train_list = [], []

        #concatenate all the other subjects
        for f in files:
            if f == test_file:
                continue
            data = all_data[f]

            X_tmp = data["X"].astype(np.float32)
            y_tmp = np.vectorize({1:0,3:1,5:2}.get)(data["y"]).astype(np.int64)
            X_tmp = X_tmp[..., np.newaxis]

            X_train_list.append(X_tmp)
            y_train_list.append(y_tmp)

        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)

        #free memory
        del X_train_list  # delete the list of individual arrays
        del y_train_list
        gc.collect()

        accs = []
        cm_total = np.zeros((3,3))

        for seed in range(n_runs):
            set_seed(seed)

            # shuffle training data each run (avoid orde bias)
            perm = np.random.permutation(len(X_train))
            X_train_s, y_train_s = X_train[perm], y_train[perm]

            #normalization
            X_train_n, X_test_n = normalize_train_test(X_train_s, X_test)

            # split TRAIN into train/val BEFORE pseudo-trials
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_n, y_train_s, test_size=0.2, stratify=y_train_s, random_state=seed
            )


            if use_pseudo_train:
                X_tr_p, y_tr_p = create_pseudo_trials(X_tr, y_tr, seed=seed)
                X_val_p, y_val_p = create_pseudo_trials(X_val, y_val, seed=seed+1)
            else:
                #no pseudo trials on train
                X_tr_p, y_tr_p = X_tr, y_tr
                X_val_p, y_val_p = X_val, y_val


            if use_pseudo_test:
                X_test_p, y_test_p = create_pseudo_trials(X_test_n, y_test, seed=seed+2)
            else:
                #no pseudo trials on test
                X_test_p, y_test_p = X_test_n, y_test   


            if downsample_mode == "match_sd":
                target_size = min(len(X_tr_p), sd_train_size)

            elif downsample_mode == "fixed":
                target_size = fixed_trials

            else:
                target_size = None

            if target_size is not None:
                X_tr_p, y_tr_p = subsample_data(
                    X_tr_p,
                    y_tr_p,
                    target_size,
                    seed
                )

            # creates the eegnet, the TCN or the CfC
            model = build_model(model_name, X_tr_p.shape[1], X_tr_p.shape[2], dropout=0.25)

            #Early stopping: stops if validation loss doesn't improve
            train_model(model, X_tr_p, y_tr_p, EPOCHS, BATCH_SIZE,X_val=X_val_p, y_val=y_val_p)

            # convert probabilities into class labeles
            preds = predict_model(model, X_test_p)
            
            #store results across runs
            accs.append(accuracy_score(y_test_p, preds))
            cm_total += confusion_matrix(y_test_p, preds, labels=[0,1,2])

            del model
            torch.cuda.empty_cache()
            gc.collect()


        #final confusion matrix: normalizes per class and make percentages
        row_sums = cm_total.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_avg = cm_total / row_sums
        per_class_acc = np.diag(cm_avg)

        mean_acc = np.mean(accs)
        accuracies.append(mean_acc)

        # store per subject
        results[test_file] = {
            "accuracy": mean_acc,
            "confusion_matrix": cm_avg,
            "per_class_accuracy": per_class_acc,
            "run_accuracies": accs
        }

        print(f"{test_file} → {mean_acc:.4f}")

    # group statistics across subject
    #results["group_stats"] = compute_statistics(accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    sem = std_acc / np.sqrt(len(accuracies))
    ci95 = 1.96 * sem

    results["group_stats"] = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "ci95": ci95,
        "n_subjects": len(accuracies)
    }
    return results


def run_transfer_learning(files, model_name, data_dir, config, n_runs=10, target_test_split=0.3, all_data=None, sd_train_size=None):

    use_pseudo_train = config["use_pseudo_train"]
    use_pseudo_test = config["use_pseudo_test"]
    downsample_mode = config["downsample_mode"]
    fixed_trials = config["fixed_trials"]

    results = {}
    accuracies = []

    for test_file in files:

        print(f"\n--- Transfer Learning on {test_file} ---")

        # Load target subject
        data = all_data[test_file]
        X = data["X"].astype(np.float32)
        y = np.vectorize({1:0,3:1,5:2}.get)(data["y"]).astype(np.int64)
        X = X[..., np.newaxis]

        X_target, y_target = X, y

        accs = []
        cm_total = np.zeros((3,3))

        for seed in range(n_runs):
            set_seed(seed)

            # Split target subject in fine-tune (to adapt the model also to this subject) + test (for evaluation)
            try:
                X_ft, X_test, y_ft, y_test = train_test_split(
                    X_target, y_target,
                    test_size=target_test_split,
                    stratify=y_target,
                    random_state=seed
                )
            except ValueError:
                X_ft, X_test, y_ft, y_test = train_test_split(
                    X_target, y_target,
                    test_size=target_test_split,
                    random_state=seed
                )


            # Build pretraining dataset (concatenate all other subjects)
            X_train_list, y_train_list = [], []

            for f in files:
                if f == test_file:
                    continue
                data = all_data[f]
                X = data["X"].astype(np.float32)
                y = np.vectorize({1:0,3:1,5:2}.get)(data["y"]).astype(np.int64)
                X = X[..., np.newaxis]

                X_tmp, y_tmp = X, y
                X_train_list.append(X_tmp)
                y_train_list.append(y_tmp)

            X_pretrain = np.concatenate(X_train_list)
            y_pretrain = np.concatenate(y_train_list)

            # free memory
            del X_train_list  # Delete the list of individual arrays
            del y_train_list
            gc.collect()

            # normalize everything using pretraining statistics
            mean = X_pretrain.mean(axis=(0,2), keepdims=True)
            std = X_pretrain.std(axis=(0,2), keepdims=True) + 1e-6
            X_pretrain_n = (X_pretrain - mean) / std
            X_ft_n = (X_ft - mean) / std
            X_test_n = (X_test - mean) / std

            # split fine-tune pool into train+val
            X_ft_train_pool, X_ft_val, y_ft_train_pool, y_ft_val  = train_test_split(
                X_ft_n, y_ft,
                test_size=0.2, 
                stratify=y_ft,
                random_state=seed
            )

            # select percentage from training pool
            fine_tune_percent = config.get("fine_tune_percent", 1.0)

            n_ft = int(len(X_ft_train_pool) * fine_tune_percent)
            perm = np.random.permutation(len(X_ft_train_pool))
            idx = perm[:n_ft]

            X_ft_tr = X_ft_train_pool[idx]
            y_ft_tr = y_ft_train_pool[idx]

            #pretrain split
            X_pre_tr, X_pre_val, y_pre_tr, y_pre_val = train_test_split(
                X_pretrain_n,
                y_pretrain,
                test_size=0.1,
                stratify=y_pretrain,
                random_state=seed
            )

            if use_pseudo_train:
                X_pre_tr_p, y_pre_tr_p = create_pseudo_trials(X_pre_tr, y_pre_tr, seed=seed)
                X_pre_val_p, y_pre_val_p = create_pseudo_trials(X_pre_val, y_pre_val, seed=seed+1)
                X_ft_tr_p, y_ft_tr_p = create_pseudo_trials(X_ft_tr, y_ft_tr, seed=seed+2)
                X_ft_val_p, y_ft_val_p = create_pseudo_trials(X_ft_val, y_ft_val, seed=seed+3)
            else:
                #no pseudo trials on train
                X_pre_tr_p, y_pre_tr_p = X_pre_tr, y_pre_tr
                X_pre_val_p, y_pre_val_p = X_pre_val, y_pre_val
                X_ft_tr_p, y_ft_tr_p = X_ft_tr, y_ft_tr
                X_ft_val_p, y_ft_val_p = X_ft_val, y_ft_val


            if use_pseudo_test:
                X_test_p, y_test_p = create_pseudo_trials(X_test_n, y_test, seed=seed+4)
            else:
                #no pseudo trials on test
                X_test_p, y_test_p = X_test_n, y_test 

            if downsample_mode == "match_sd":
                target_size = min(len(X_pre_tr_p), sd_train_size)

            elif downsample_mode == "fixed":
                target_size = fixed_trials

            else:
                target_size = None

            if target_size is not None:
                X_pre_tr_p, y_pre_tr_p = subsample_data(
                    X_pre_tr_p,
                    y_pre_tr_p,
                    target_size,
                    seed
                )

            # Build model
            model = build_model(model_name, X_pre_tr_p.shape[1], X_pre_tr_p.shape[2], dropout=0.25)

            #Pretraining: learn general patterns 
            train_model(model, X_pre_tr_p, y_pre_tr_p, EPOCHS_PRETRAIN, BATCH_SIZE, X_val=X_pre_val_p, y_val=y_pre_val_p)

            # FINE-TUNING
            # freeze all layers except final FC
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False


            # new optimizer (only trainable params)
            optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

            train_model(
                model, 
                X_ft_tr_p, y_ft_tr_p, 
                EPOCHS, BATCH_SIZE, 
                X_val=X_ft_val_p, y_val=y_ft_val_p,
                optimizer=optimizer
            )

            
            #test convert probabilities into class labeles
            preds = predict_model(model, X_test_p)

            #store results across runs
            accs.append(accuracy_score(y_test_p, preds))
            cm_total += confusion_matrix(y_test_p, preds, labels=[0,1,2])

            #Cleanup
            del model
            torch.cuda.empty_cache()
            

        #final confusion matrix: normalizes per class and make percentages
        row_sums = cm_total.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_avg = cm_total / row_sums
        per_class_acc = np.diag(cm_avg)

        mean_acc = np.mean(accs)
        accuracies.append(mean_acc)

        # store per subject
        results[test_file] = {
            "accuracy": mean_acc,
            "confusion_matrix": cm_avg,
            "per_class_accuracy": per_class_acc,
            "run_accuracies": accs
        }

        print(f"{test_file} → {mean_acc:.4f}")
        
    #group statistics across subject
    #results["group_stats"] = compute_statistics(accuracies)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    sem = std_acc / np.sqrt(len(accuracies))
    ci95 = 1.96 * sem

    results["group_stats"] = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "ci95": ci95,
        "n_subjects": len(accuracies)
    }

    return results