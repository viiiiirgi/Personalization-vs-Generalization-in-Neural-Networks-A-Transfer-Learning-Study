import numpy as np
import pandas as pd
import os
import gc

CSV_FILES = {
    "BSL": "EEGtrial_cleanEEGMAT_BSL.csv",
    "DELAY": "EEGtrial_cleanEEGMAT_DELAY.csv",
    "SENSORY": "EEGtrial_cleanEEGMAT_SENSORY.csv",
}

OUTPUT_DIR = "subject_npy"
CHUNK_SIZE = 100_000
DTYPE = np.float32

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_unique_subjects(csv_file):

    subjects = set()

    for chunk in pd.read_csv(
        csv_file,
        header=None,
        chunksize=CHUNK_SIZE
    ):
        subjects.update(chunk.iloc[:, -2].dropna().unique())

    return sorted(subjects)



def load_one_subject(csv_file, subject_id):

    subject_chunks = []

    for chunk in pd.read_csv(
        csv_file,
        header=None,
        dtype=DTYPE,
        chunksize=CHUNK_SIZE
    ):
        filtered = chunk[chunk.iloc[:, -2] == subject_id]

        if not filtered.empty:
            subject_chunks.append(filtered)

    if len(subject_chunks) == 0:
        return None

    data = pd.concat(subject_chunks, axis=0).values

    del subject_chunks
    gc.collect()

    return data



def reconstruct_trials(data):

    X_time = data[:, :-4]
    labels = data[:, -4].astype(int)
    channels = data[:, -3].astype(int)
    trials = data[:, -1].astype(int)

    unique_trials = np.unique(trials)
    unique_channels = np.unique(channels)

    n_trials = len(unique_trials)
    n_channels = len(unique_channels)
    n_times = X_time.shape[1]

    X = np.zeros((n_trials, n_channels, n_times), dtype=DTYPE)
    y = np.zeros(n_trials, dtype=int)

    trial_index = {t: i for i, t in enumerate(unique_trials)}
    channel_index = {c: i for i, c in enumerate(unique_channels)}

    for row in range(X_time.shape[0]):
        ti = trial_index[trials[row]]
        ci = channel_index[channels[row]]

        X[ti, ci, :] = X_time[row]
        y[ti] = labels[row]

    return X, y



def convert_condition(condition_name, csv_file):

    print(f"\n=== Converting {condition_name} ===")

    subjects = get_unique_subjects(csv_file)

    print(f"Found {len(subjects)} subjects")

    for i, sub in enumerate(subjects):

        print(f"[{i+1}/{len(subjects)}] Subject {sub}")

        data = load_one_subject(csv_file, sub)

        if data is None:
            continue

        X, y = reconstruct_trials(data)

        save_path = os.path.join(
            OUTPUT_DIR,
            f"{condition_name}_subject_{int(sub)}.npy"
        )

        np.save(save_path, {"X": X, "y": y})

        print(f"Saved → {save_path}")
        print("Shape:", X.shape)

        del data, X, y
        gc.collect()



if __name__ == "__main__":

    for cond, csv_file in CSV_FILES.items():
        convert_condition(cond, csv_file)

    print("\nAll conversions complete.")
