import os
import numpy as np
from experiments.experiments_eegnet import (
    run_subject_dependent,
    run_subject_independent,
    run_transfer_learning
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL = "eegnet"   # "eegnet", "tcn", "cfc"
MODE = "SD"      # "SD", "SI", "TL", "ALL"
CONDITIONS = ["BSL", "DELAY", "SENSORY"]

def main():

    for CONDITION in CONDITIONS:
        
        print(f"Running condition: {CONDITION}")

        save_dir = os.path.join(RESULTS_DIR, MODEL)
        os.makedirs(save_dir, exist_ok=True)

        files = sorted(
            f for f in os.listdir(DATA_DIR)
            if f.endswith(".npy") and f.startswith(CONDITION + "_")
        )

        if MODE in ["SD", "ALL"]:
            print("\n===== SUBJECT-DEPENDENT =====")

            sd_results = {}
            accs = []

            for f in files:
                acc, cm = run_subject_dependent(f"{DATA_DIR}/{f}", MODEL)
                sd_results[f] = {"accuracy": acc, "confusion_matrix": cm}
                accs.append(acc)

            np.save(f"{save_dir}/{CONDITION}_sd.npy", sd_results)

        if MODE in ["SI", "ALL"]:
            print("\n===== SUBJECT-INDEPENDENT =====")

            si_results = run_subject_independent(files, MODEL, DATA_DIR)

            np.save(f"{save_dir}/{CONDITION}_si.npy", si_results)
        
        if MODE in ["TL", "ALL"]:
            print("\n===== TRANSFER LEARNING =====")

            tl_results = run_transfer_learning(files, MODEL, DATA_DIR)

            np.save(f"{save_dir}/{CONDITION}_tl.npy", tl_results)

if __name__ == "__main__":
    main()