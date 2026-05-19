print("!!! SCRIPT INITIALIZED !!!")
import os
import numpy as np
from experiments.experiments_eegnet import (
    run_subject_dependent,
    run_subject_independent,
    run_transfer_learning
)
from utils.train_utils import compute_sd_train_size
from utils.train_utils import set_seed
set_seed(1) 


BASE_DIR = BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL = "tcn"   # "eegnet", "tcn", "cfc" 
MODE = "SI"      # "SD", "SI", "TL", "ALL"
CONDITIONS = ["BSL", "DELAY", "SENSORY"] #["BSL", "DELAY", "SENSORY"]

def main():
    
    for CONDITION in CONDITIONS:
        
        print(f"Running condition: {CONDITION}")

        save_dir = os.path.join(RESULTS_DIR, MODEL)
        os.makedirs(save_dir, exist_ok=True)

        files = sorted(
            f for f in os.listdir(DATA_DIR)
            if f.endswith(".npy") and f.startswith(CONDITION + "_")
        )

        print("Preloading all data into RAM...")
        all_data = {
            f: np.load(f"{DATA_DIR}/{f}", allow_pickle=True).item()
            for f in files
        }

        sd_train_size = compute_sd_train_size(files, all_data)
        print("SD_TRAIN_SIZE:", sd_train_size)

        if MODE in ["SD", "ALL"]:
            print("\n===== SUBJECT-DEPENDENT =====")

            sd_results = {}
            accs = []

            for f in files:
                acc, cm = run_subject_dependent(f"{DATA_DIR}/{f}", MODEL, all_data=all_data)
                sd_results[f] = {"accuracy": acc, "confusion_matrix": cm}
                accs.append(acc)
                print(f"{f} → {acc:.4f}")

            np.save(f"{save_dir}/{CONDITION}_sd.npy", sd_results)

        if MODE in ["SI", "ALL"]:
            print("\n===== SUBJECT-INDEPENDENT =====")

            si_results = run_subject_independent(files, MODEL, DATA_DIR, all_data=all_data, sd_train_size=sd_train_size)

            np.save(f"{save_dir}/{CONDITION}_si.npy", si_results)
        
        if MODE in ["TL", "ALL"]:
            print("\n===== TRANSFER LEARNING =====")

            tl_results = run_transfer_learning(files, MODEL, DATA_DIR, all_data=all_data, sd_train_size=sd_train_size)

            np.save(f"{save_dir}/{CONDITION}_tl.npy", tl_results)

if __name__ == "__main__":
    main()