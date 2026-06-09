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

try:
    from utils.feature_viz import export_model_visualizations, subject_viz_prefix
except ImportError:
    export_model_visualizations = None
    subject_viz_prefix = lambda f: f.replace(".npy", "").split("_", 1)[-1]

EPOCHS_PRETRAIN = 20
EPOCHS = 30
BATCH_SIZE = 64

EXPERIMENTS = {

    "AllData_NoPseudo": {
        "experiment": "AllData_NoPseudo",
        "use_pseudo_train": False,
        "use_pseudo_test": False,
        "downsample_mode": None,
        "fixed_trials": None,
    },

    "AllData_PseudoTrainAndTest": {
        "experiment": "AllData_PseudoTrainAndTest",
        "use_pseudo_train": True,
        "use_pseudo_test": True,
        "downsample_mode": None,
        "fixed_trials": None,
    },

    "AllData_PseudoTrainOnly": {
        "experiment": "AllData_PseudoTrainOnly",
        "use_pseudo_train": True,
        "use_pseudo_test": False,
        "downsample_mode": None,
        "fixed_trials": None,
    },

    "Downsample_1000_NoPseudo": {
        "experiment": "Downsample_1000_NoPseudo",
        "use_pseudo_train": False,
        "use_pseudo_test": False,
        "downsample_mode": "fixed",
        "fixed_trials": 1000,
    },

    "Downsample_MinSubject_NoPseudo": {
        "experiment": "Downsample_MinSubject_NoPseudo",
        "use_pseudo_train": False,
        "use_pseudo_test": False,
        "downsample_mode": "match_sd",
        "fixed_trials": None,
    }
}

FINE_TUNE_PERCENTS = [0.1, 0.25, 0.5, 0.75, 1.0]


# train and test on the same subject
def run_subject_dependent(file, model_name, config, n_runs=10, all_data=None, viz_dir=None, viz_seed=0):

    use_pseudo_train = config["use_pseudo_train"]
    use_pseudo_test = config["use_pseudo_test"]
    experiment = config["experiment"]

    if all_data is not None:
        # extract filename only (because keys are filenames, not full paths)
        key = os.path.basename(file)
        data = all_data[key]
    else:
        data = np.load(file, allow_pickle=True).item()

    X = data["X"].astype(np.float32)
    y = np.vectorize({1:0,3:1,5:2}.get)(data["y"]).astype(np.int64)
    X = X[..., np.newaxis] #X=EEG data (trials x channel x time), y=labeles

    accs = [] 
    cm_total = np.zeros((3,3)) #confusion matrices
    best_acc= -1
    sustainability_runs = []

    for seed in range(n_runs):
        set_seed(seed)

        # random splits
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

        # normalization
        X_train_n, X_test_n = normalize_train_test(X_train, X_test)
        
        # split TRAIN into train/val BEFORE pseudo-trials
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_n, y_train, test_size=0.2, stratify=y_train, random_state=seed)

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
        model, metrics= train_model(model, X_tr_p, y_tr_p, EPOCHS, BATCH_SIZE, X_val=X_val_p, y_val=y_val_p)

        
        # convert probabilities into class labeles
        preds = predict_model(model, X_test_p)

        # optional: save feature plots at seed 0 (does not change accuracy)
        if viz_dir is not None and seed == viz_seed and export_model_visualizations is not None:
            subj_prefix = subject_viz_prefix(os.path.basename(file))
            export_model_visualizations(model, X_test_p, y_test_p, viz_dir, f"{subj_prefix}_sd")

        #store results across runs
        acc = accuracy_score(y_test_p, preds)
        accs.append(acc)
        sustainability_runs.append(metrics)

        #save best model
        if seed == 0 or acc > best_acc:
            best_acc = acc

            save_dir = os.path.join("saved_models", experiment)
            os.makedirs(save_dir, exist_ok=True)

            torch.save(model.state_dict(),os.path.join( save_dir, f"sd_{os.path.basename(file)}_best.pt"))

        cm_total += confusion_matrix(y_test_p, preds, labels=[0,1,2])

        del model
        torch.cuda.empty_cache()

    #average sustainibility metrics 
    mean_metrics = {
        "training_time_s":
            np.mean([m["training_time_s"] for m in sustainability_runs]),
        "peak_ram_mb":
            np.mean([m["peak_ram_mb"] for m in sustainability_runs]),
        "peak_gpu_mb":
            np.mean([m["peak_gpu_mb"] for m in sustainability_runs]),
        "energy_kwh":
            np.mean([m["energy_kwh"] for m in sustainability_runs]),
        "co2_kg":
            np.mean([m["co2_kg"] for m in sustainability_runs]),
    }

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

    results["sustainability_stats"] = {
        "mean_training_time_s": mean_metrics["training_time_s"],
        "mean_peak_ram_mb": mean_metrics["peak_ram_mb"],
        "mean_peak_gpu_mb": mean_metrics["peak_gpu_mb"],
        "mean_energy_kwh": mean_metrics["energy_kwh"],
        "mean_co2_kg": mean_metrics["co2_kg"]
    }

    #return np.mean(accs), cm_avg
    return results

# train on all subjects except one, test on that one
def run_subject_independent(files, model_name, data_dir, config, n_runs=10, all_data=None, sd_train_size=None,
                            viz_dir=None, viz_target=None, viz_seed=0):

    use_pseudo_train = config["use_pseudo_train"]
    use_pseudo_test = config["use_pseudo_test"]
    downsample_mode = config["downsample_mode"]
    fixed_trials = config["fixed_trials"]
    experiment = config["experiment"]

    results = {}
    accuracies = []

    for test_file in files: # leave one subject out

        sustainability_runs = []

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
        best_acc=-1

        for seed in range(n_runs):
            set_seed(seed)

            # shuffle training data each run (avoid orde bias)
            perm = np.random.permutation(len(X_train))
            X_train_s, y_train_s = X_train[perm], y_train[perm]

            #normalization
            X_train_n, X_test_n = normalize_train_test(X_train_s, X_test)

            # split train into train/val before pseudo-trials
            X_tr, X_val, y_tr, y_val = train_test_split(X_train_n, y_train_s, test_size=0.2, stratify=y_train_s, random_state=seed)

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
                X_tr_p, y_tr_p = subsample_data(X_tr_p, y_tr_p, target_size, seed)

            # creates the eegnet, the TCN or the CfC
            model = build_model(model_name, X_tr_p.shape[1], X_tr_p.shape[2], dropout=0.25)

            #Early stopping: stops if validation loss doesn't improve
            model, metrics = train_model(model, X_tr_p, y_tr_p, EPOCHS, BATCH_SIZE,X_val=X_val_p, y_val=y_val_p)

            # convert probabilities into class labeles
            preds = predict_model(model, X_test_p)

            # optional: save feature plots at seed 0 for the viz target subject
            if (viz_dir is not None and seed == viz_seed and viz_target is not None and test_file == viz_target
                                    and export_model_visualizations is not None):
                subj_prefix = subject_viz_prefix(test_file)
                export_model_visualizations(model, X_test_p, y_test_p, viz_dir, f"{subj_prefix}_si")

            #store results across runs
            acc = accuracy_score(y_test_p, preds)
            accs.append(acc)
            sustainability_runs.append(metrics)

            if seed == 0 or acc > best_acc:
                best_acc = acc

                save_dir = os.path.join("saved_models", experiment)
                os.makedirs(save_dir, exist_ok=True)

                torch.save(model.state_dict(), os.path.join( save_dir, f"si_{os.path.basename(test_file)}_best.pt"))

            cm_total += confusion_matrix(y_test_p, preds, labels=[0,1,2])

            del model
            torch.cuda.empty_cache()
            gc.collect()

        #mean sustainability metrics
        mean_metrics = {
            "training_time_s":
                np.mean([m["training_time_s"] for m in sustainability_runs]),
            "peak_ram_mb":
                np.mean([m["peak_ram_mb"] for m in sustainability_runs]),
            "peak_gpu_mb":
                np.mean([m["peak_gpu_mb"] for m in sustainability_runs]),
            "energy_kwh":
                np.mean([m["energy_kwh"] for m in sustainability_runs]),
            "co2_kg":
                np.mean([m["co2_kg"] for m in sustainability_runs]),
        }

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
            "run_accuracies": accs,
            "sustainability": mean_metrics
        }

        print(f"{test_file} → {mean_acc:.4f}")

    # group statistics across subject
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

    all_training_times = [
        results[s]["sustainability"]["training_time_s"]
        for s in results
        if s != "group_stats"
    ]
    all_energy = [
        results[s]["sustainability"]["energy_kwh"]
        for s in results
        if s != "group_stats"
    ]
    all_co2 = [
        results[s]["sustainability"]["co2_kg"]
        for s in results
        if s != "group_stats"
    ]
    all_ram = [
        results[s]["sustainability"]["peak_ram_mb"]
        for s in results
        if s != "group_stats"
    ]
    all_gpu = [
        results[s]["sustainability"]["peak_gpu_mb"]
        for s in results
        if s != "group_stats"
    ]

    results["sustainability_stats"] = {
        "mean_training_time_s": np.mean(all_training_times),
        "mean_peak_ram_mb": np.mean(all_ram),
        "mean_peak_gpu_mb": np.mean(all_gpu),
        "mean_energy_kwh": np.mean(all_energy),
        "mean_co2_kg": np.mean(all_co2)
    }

    return results


def run_transfer_learning(files, model_name, data_dir, config, n_runs=10, target_test_split=0.3, all_data=None, sd_train_size=None,
                          viz_dir=None, viz_target=None, viz_seed=0):

    use_pseudo_train = config["use_pseudo_train"]
    use_pseudo_test = config["use_pseudo_test"]
    downsample_mode = config["downsample_mode"]
    fixed_trials = config["fixed_trials"]
    experiment = config["experiment"]

    results = {}
    accuracies = []

    for test_file in files:

        print(f"\n--- Transfer Learning on {test_file} ---")

        sustainability_runs = []

        # Load target subject
        data = all_data[test_file]
        X = data["X"].astype(np.float32)
        y = np.vectorize({1:0,3:1,5:2}.get)(data["y"]).astype(np.int64)
        X = X[..., np.newaxis]

        X_target, y_target = X, y

        accs = []
        cm_total = np.zeros((3,3))
        best_acc=-1

        for seed in range(n_runs):
            set_seed(seed)

            # Split target subject in fine-tune (to adapt the model also to this subject) + test (for evaluation)
            try:
                X_ft, X_test, y_ft, y_test = train_test_split(X_target, y_target, test_size=target_test_split, stratify=y_target, random_state=seed)
            except ValueError:
                X_ft, X_test, y_ft, y_test = train_test_split(X_target, y_target, test_size=target_test_split, random_state=seed)


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
            X_ft_train_pool, X_ft_val, y_ft_train_pool, y_ft_val  = train_test_split(X_ft_n, y_ft, test_size=0.2, stratify=y_ft, random_state=seed)

            # select percentage from training pool
            fine_tune_percent = config.get("fine_tune_percent", 1.0)

            n_ft = int(len(X_ft_train_pool) * fine_tune_percent)
            perm = np.random.permutation(len(X_ft_train_pool))
            idx = perm[:n_ft]

            X_ft_tr = X_ft_train_pool[idx]
            y_ft_tr = y_ft_train_pool[idx]

            #pretrain split
            X_pre_tr, X_pre_val, y_pre_tr, y_pre_val = train_test_split(X_pretrain_n, y_pretrain, test_size=0.1, stratify=y_pretrain, random_state=seed)

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
                X_pre_tr_p, y_pre_tr_p = subsample_data( X_pre_tr_p, y_pre_tr_p, target_size, seed)

            # Build model
            model = build_model(model_name, X_pre_tr_p.shape[1], X_pre_tr_p.shape[2], dropout=0.25)

            #Pretraining: learn general patterns 
            model, pre_metrics = train_model(model, X_pre_tr_p, y_pre_tr_p, EPOCHS_PRETRAIN, BATCH_SIZE, X_val=X_pre_val_p, y_val=y_pre_val_p)

            # FINE-TUNING
            # freeze all layers except final FC
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False


            # new optimizer (only trainable params)
            optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

            model, ft_metrics= train_model(model, X_ft_tr_p, y_ft_tr_p, EPOCHS, BATCH_SIZE, X_val=X_ft_val_p, y_val=y_ft_val_p, optimizer=optimizer)
            
            #combine sustainability metrics
            metrics = {
                "training_time_s":
                    pre_metrics["training_time_s"] + ft_metrics["training_time_s"],
                "peak_ram_mb":
                    max(pre_metrics["peak_ram_mb"],ft_metrics["peak_ram_mb"]),
                "peak_gpu_mb":
                    max(pre_metrics["peak_gpu_mb"],ft_metrics["peak_gpu_mb"]),
                "energy_kwh":
                    pre_metrics["energy_kwh"] + ft_metrics["energy_kwh"],
                "co2_kg":
                    pre_metrics["co2_kg"] + ft_metrics["co2_kg"]
            }

            #test convert probabilities into class labeles
            preds = predict_model(model, X_test_p)

            # optional: save feature plots at seed 0 for the viz target subject
            if (viz_dir is not None and seed == viz_seed and viz_target is not None and test_file == viz_target 
                                    and export_model_visualizations is not None):
                subj_prefix = subject_viz_prefix(test_file)
                export_model_visualizations(model, X_test_p, y_test_p, viz_dir, f"{subj_prefix}_tl")

            #store results across runs
            acc = accuracy_score(y_test_p, preds)
            accs.append(acc)
            sustainability_runs.append(metrics)

            if seed == 0 or acc > best_acc:
                best_acc = acc

                save_dir = os.path.join("saved_models", experiment)
                os.makedirs(save_dir, exist_ok=True)

                torch.save(model.state_dict(), os.path.join( save_dir, f"tl_{os.path.basename(test_file)}_best.pt"))

            cm_total += confusion_matrix(y_test_p, preds, labels=[0,1,2])

            #Cleanup
            del model
            torch.cuda.empty_cache()
            
        #mean sustainability metrics
        mean_metrics = {
            "training_time_s":
                np.mean([m["training_time_s"] for m in sustainability_runs]),
            "peak_ram_mb":
                np.mean([m["peak_ram_mb"] for m in sustainability_runs]),
            "peak_gpu_mb":
                np.mean([m["peak_gpu_mb"] for m in sustainability_runs]),
            "energy_kwh":
                np.mean([m["energy_kwh"] for m in sustainability_runs]),
            "co2_kg":
                np.mean([m["co2_kg"] for m in sustainability_runs]),
        }

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
            "run_accuracies": accs,
            "sustainability": mean_metrics
        }

        print(f"{test_file} → {mean_acc:.4f}")
        

    #group statistics across subject
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

    all_training_times = [
        results[s]["sustainability"]["training_time_s"]
        for s in results
        if s != "group_stats"
    ]
    all_energy = [
        results[s]["sustainability"]["energy_kwh"]
        for s in results
        if s != "group_stats"
    ]
    all_co2 = [
        results[s]["sustainability"]["co2_kg"]
        for s in results
        if s != "group_stats"
    ]
    all_ram = [
        results[s]["sustainability"]["peak_ram_mb"]
        for s in results
        if s != "group_stats"
    ]
    all_gpu = [
        results[s]["sustainability"]["peak_gpu_mb"]
        for s in results
        if s != "group_stats"
    ]

    results["sustainability_stats"] = {
        "mean_training_time_s": np.mean(all_training_times),
        "mean_peak_ram_mb": np.mean(all_ram),
        "mean_peak_gpu_mb": np.mean(all_gpu),
        "mean_energy_kwh": np.mean(all_energy),
        "mean_co2_kg": np.mean(all_co2)
    }

    return results