import os
import numpy as np
import torch
import gc
from experiments.experiments_eegnet import (
    run_subject_dependent,
    run_subject_independent,
    run_transfer_learning,
    EXPERIMENTS,
    FINE_TUNE_PERCENTS
)
from utils.train_utils import compute_sd_train_size, set_seed

set_seed(1) 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL = "eegnet"   # Options: "eegnet", "tcn", "cfc" 
MODE = "ALL"       # Runs "SD", "SI", and "TL" sequentially
CONDITIONS = ["BSL", "DELAY", "SENSORY"]

def main():
    for CONDITION in CONDITIONS:
        print(f"\nRunning condition: {CONDITION}")

        files = sorted(
            f for f in os.listdir(DATA_DIR)
            if f.endswith(".npy") and f.startswith(CONDITION + "_")
        )

        print("Preloading all data into RAM...")
        all_data = {
            f: np.load(os.path.join(DATA_DIR, f), allow_pickle=True).item()
            for f in files
        }

        sd_train_size = compute_sd_train_size(files, all_data)
        print("SD_TRAIN_SIZE:", sd_train_size)

        for exp_name, config in EXPERIMENTS.items():
            print(f"\n==============================")
            print(f"RUNNING EXPERIMENT: {exp_name}")
            print(f"==============================")

            save_dir = os.path.join(RESULTS_DIR, MODEL, exp_name)
            os.makedirs(save_dir, exist_ok=True)

           
            if MODE in ["SD", "ALL"]:
                print(f"[{exp_name}] Running Subject-Dependent Pipeline...")
                sd_results = {}

                for f in files:
                    subj_results = run_subject_dependent(
                        os.path.join(DATA_DIR, f), 
                        MODEL, 
                        config=config, 
                        all_data=all_data
                    )
                    sd_results[f] = subj_results
                    print(f"{f} → {subj_results['accuracy']:.4f}")

                np.save(os.path.join(save_dir, f"{CONDITION}_sd.npy"), sd_results)

           
            if MODE in ["SI", "ALL"]:
                print(f"[{exp_name}] Running Subject-Independent Pipeline...")
                si_results = run_subject_independent(
                    files, 
                    MODEL, 
                    DATA_DIR, 
                    config=config, 
                    all_data=all_data, 
                    sd_train_size=sd_train_size
                )
                np.save(os.path.join(save_dir, f"{CONDITION}_si.npy"), si_results)
            
           
            if MODE in ["TL", "ALL"]:
                print(f"[{exp_name}] Running Transfer Learning Pipeline...")

                if exp_name == "AllData_NoPseudo":
                    for ft_percent in FINE_TUNE_PERCENTS:
                        config_ft = config.copy()
                        config_ft["fine_tune_percent"] = ft_percent
                        absolute_dataset_percent = int(round(ft_percent * 70.0))

                        ft_dir = os.path.join(save_dir, f"FT_{absolute_dataset_percent}")
                        os.makedirs(ft_dir, exist_ok=True)

                        tl_results = run_transfer_learning(
                            files,
                            MODEL,
                            DATA_DIR,
                            config=config_ft,
                            all_data=all_data,
                            sd_train_size=sd_train_size
                        )

                        if ft_percent == 1.0:
                            np.save(os.path.join(save_dir, f"{CONDITION}_tl.npy"),tl_results)
                        
                        np.save(os.path.join(ft_dir, f"{CONDITION}_tl.npy"), tl_results)
                
                else: 
                    tl_results = run_transfer_learning(
                        files,
                        MODEL,
                        DATA_DIR,
                        config=config,
                        all_data=all_data,
                        sd_train_size=sd_train_size
                    )
                    np.save(os.path.join(save_dir, f"{CONDITION}_tl.npy"), tl_results)

            # Clean up GPU memory and call garbage collection after each experiment setup
            torch.cuda.empty_cache()
            gc.collect()
        
        # Clear out data tracking array before loading the next environment condition
        del all_data
        gc.collect()

if __name__ == "__main__":
    main()