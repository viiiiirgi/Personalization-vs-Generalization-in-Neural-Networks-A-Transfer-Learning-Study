import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import ttest_1samp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from utils.data_loader import load_subject
from utils.train_utils import (
    set_seed,
    normalize_train_test,
    create_pseudo_trials,
    build_model,
    clear
)
import gc

EPOCHS = 10
BATCH_SIZE = 32


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

# train and test on the same subject
def run_subject_dependent(file, model_name, n_runs=10):

    X, y = load_subject(file) #X=EEG data (trials x channel x time), y=labeles

    chans = X.shape[1]
    samples = X.shape[2]

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

        #normalization
        X_train_n, X_test_n = normalize_train_test(X_train, X_test)

        #pseudo trials (averages groups of trials reducing noise)
        X_train_p, y_train_p = create_pseudo_trials(X_train_n, y_train)
        X_test_p, y_test_p = create_pseudo_trials(X_test_n, y_test)

        # creates the eegnet, the TCN or the CfC
        model = build_model(model_name, chans, samples)

        #Early stopping: stops if validation loss doesn't improve
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        model.fit(
                      X_train_p, y_train_p,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      verbose=0,            
                      validation_split=0.2, #internal validation
                      callbacks=callbacks
                  )
        
        # convert probabilities into class labeles
        preds = np.argmax(model.predict(X_test_p), axis=1)

        #store results across runs
        accs.append(accuracy_score(y_test_p, preds))
        cm_total += confusion_matrix(y_test_p, preds, labels=[0,1,2])

        #free gpu
        clear()

    #final confusion matrix: normalizes per class and make percentages
    row_sums = cm_total.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_avg = cm_total / row_sums

    #final subject performance
    return np.mean(accs), cm_avg

# train on all subjects except one, test on that one
def run_subject_independent(files, model_name, data_dir, n_runs=10):

    results = {}
    accuracies = []

    for test_file in files: # leave one subject out

        # select test subject
        X_test, y_test = load_subject(f"{data_dir}/{test_file}")

        X_train_list, y_train_list = [], []

        #concatenate all the other subjects
        for f in files:
            if f == test_file:
                continue
            X_tmp, y_tmp = load_subject(f"{data_dir}/{f}")
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

            #pseudo trials (averages groups of trials reducing noise)
            X_train_p, y_train_p = create_pseudo_trials(X_train_n, y_train_s)
            X_test_p, y_test_p = create_pseudo_trials(X_test_n, y_test)

            # creates the eegnet, the TCN or the CfC
            model = build_model(model_name, X_train.shape[1], X_train.shape[2])

            #Early stopping: stops if validation loss doesn't improve
            callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
            model.fit(
                      X_train_p, y_train_p,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      verbose=0,
                      validation_split=0.2,
                      callbacks=callbacks
                  )

            # convert probabilities into class labeles
            preds = np.argmax(model.predict(X_test_p), axis=1)

            #store results across runs
            accs.append(accuracy_score(y_test_p, preds))
            cm_total += confusion_matrix(y_test_p, preds, labels=[0,1,2])

            #free gpu
            clear()

        #final confusion matrix: normalizes per class and make percentages
        row_sums = cm_total.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_avg = cm_total / row_sums

        mean_acc = np.mean(accs)
        accuracies.append(mean_acc)

        # store per subject
        results[test_file] = {
            "accuracy": mean_acc,
            "confusion_matrix": cm_avg
        }

        print(f"{test_file} → {mean_acc:.4f}")

    # group statistics across subject
    results["group_stats"] = compute_statistics(accuracies)
    return results


def run_transfer_learning(files, model_name, data_dir, n_runs=10, fine_tune_split=0.3):

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

            # Split target subject in fine-tune (to adapt the model also to this subject) + test (for evaluation)
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

            # Build pretraining dataset (concatenate all other subjects)
            X_train_list, y_train_list = [], []

            for f in files:
                if f == test_file:
                    continue
                X_tmp, y_tmp = load_subject(f"{data_dir}/{f}")
                X_train_list.append(X_tmp)
                y_train_list.append(y_tmp)

            X_pretrain = np.concatenate(X_train_list)
            y_pretrain = np.concatenate(y_train_list)

            # free memory
            del X_train_list  # Delete the list of individual arrays
            del y_train_list
            gc.collect()

            # Normalization using the pretraining stats (avoids leakage from target subject)
            X_pretrain_n, X_test_n = normalize_train_test(X_pretrain, X_test)
            _, X_ft_n = normalize_train_test(X_pretrain, X_ft)

            # Pseudo trials (averages groups of trials reducing noise) (applied to pretraining, fine-tuning and testing)
            X_pre_p, y_pre_p = create_pseudo_trials(X_pretrain_n, y_pretrain)
            X_ft_p, y_ft_p = create_pseudo_trials(X_ft_n, y_ft)
            X_test_p, y_test_p = create_pseudo_trials(X_test_n, y_test)

            # Build model
            model = build_model(model_name, X_pretrain.shape[1], X_pretrain.shape[2])

            #Early stopping: stops if validation loss doesn't improve
            callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
            #Pretraining: learn general patterns 
            model.fit(
                          X_pre_p, y_pre_p,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          verbose=0,
                          validation_split=0.2,
                          callbacks=callbacks
                      )


            #FINE-TUNING
            #freeze all the layers except from the classifier (keep larned features and update classifier)
            for layer in model.layers[:-2]: 
                layer.trainable = False  

            # rebuild the training graph and know which weights are trainable after freezing the layers
            model.compile(  
                optimizer=Adam(1e-4),  # recompile with lower learning rate for fine-tuning
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
            #adapt to a specific subject
            model.fit(
                          X_ft_p, y_ft_p,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          verbose=0,
                          validation_split=0.2,
                          callbacks=callbacks
                      )


            # convert probabilities into class labeles
            preds = np.argmax(model.predict(X_test_p), axis=1)

            #store results across runs
            accs.append(accuracy_score(y_test_p, preds))
            cm_total += confusion_matrix(y_test_p, preds, labels=[0,1,2])
            
            #free gpu
            clear()

        #final confusion matrix: normalizes per class and make percentages
        row_sums = cm_total.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_avg = cm_total / row_sums

        mean_acc = np.mean(accs)
        accuracies.append(mean_acc)
        
        # store per subject
        results[test_file] = {
            "accuracy": mean_acc,
            "confusion_matrix": cm_avg
        }

        print(f"{test_file} → {mean_acc:.4f}")
        
    #group statistics across subject
    results["group_stats"] = compute_statistics(accuracies)
    return results