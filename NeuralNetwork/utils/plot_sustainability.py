import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL = "eegnet"
CONDITION = "SENSORY"  
FT_PERCENTS = [10, 25, 50, 75, 100]

EXPERIMENT_LIST = [
    "AllData_NoPseudo",
    "AllData_PseudoTrainAndTest",
    "AllData_PseudoTrainOnly",
    "Downsample_1000_NoPseudo",
    "Downsample_MinSubject_NoPseudo"
]

PARADIGM_STYLES = {
    "SD": {"color": "#4A90E2", "linestyle": "--", "label": "Subject-Dependent Baseline (SD)"},
    "SI": {"color": "#E67E22", "linestyle": "-.", "label": "Subject-Independent Baseline (SI)"},
    "TL": {"color": "#D5A6E6", "marker": "^", "linestyle": "-", "label": "Transfer Learning Curve (TL)"}
}

metrics_to_plot = [
    {"key": "mean_co2_kg", "title": "Carbon Footprint", "ylabel": "CO2 Emissions (kg)"},
    {"key": "mean_energy_kwh", "title": "Energy Consumption", "ylabel": "Energy (kWh)"},
    {"key": "mean_training_time_s", "title": "Compute Time", "ylabel": "Training Duration (s)"},
    {"key": "mean_peak_gpu_mb", "title": "Peak VRAM Allocation", "ylabel": "GPU Memory (MB)"}
]

for experiment in EXPERIMENT_LIST:
    print(f"\nProcessing Experiment Viz: {experiment}...")
    
    current_exp_dir = os.path.join(RESULTS_DIR, MODEL, experiment)
    if not os.path.exists(current_exp_dir):
        print(f"Skipping: Directory not found for {experiment}")
        continue

    # 1. Extract standard fixed baseline metrics (SD, SI) from the root directory
    baselines = {"SD": {}, "SI": {}}
    for paradigm in ["SD", "SI"]:
        root_file = os.path.join(current_exp_dir, f"{CONDITION}_{paradigm.lower()}.npy")
        if os.path.exists(root_file):
            try:
                data = np.load(root_file, allow_pickle=True).item()
                sust = data.get("sustainability_stats", {})
                for m in metrics_to_plot:
                    baselines[paradigm][m["key"]] = sust.get(m["key"], 0)
            except Exception as e:
                print(f"Error loading baseline {paradigm}: {e}")

    # 2. Check how Transfer Learning data is structured for this experiment
    tl_x = []
    tl_stats = {m["key"]: [] for m in metrics_to_plot}
    
    # Test if fractional subdirectories exist by checking for FT_100 or FT_70
    has_fraction_folders = os.path.exists(os.path.join(current_exp_dir, "FT_100")) or \
                           os.path.exists(os.path.join(current_exp_dir, "FT_70"))

    if has_fraction_folders:
        # Dynamic Fractional Data Path Parsing (e.g., AllData_NoPseudo scenario)
        print("  -> Found fractional subdirectories. Extracting curve...")
        for p in FT_PERCENTS:
            fraction = p / 100.0
            absolute_percent = int(round(fraction * 70.0))
            
            tl_file_A = os.path.join(current_exp_dir, f"FT_{absolute_percent}", f"{CONDITION}_tl.npy")
            tl_file_B = os.path.join(current_exp_dir, f"FT_{p}", f"{CONDITION}_tl.npy")
            chosen_tl_file = tl_file_A if os.path.exists(tl_file_A) else (tl_file_B if os.path.exists(tl_file_B) else None)
            
            if chosen_tl_file:
                try:
                    data = np.load(chosen_tl_file, allow_pickle=True).item()
                    sust = data.get("sustainability_stats", {})
                    tl_x.append(p)
                    for m in metrics_to_plot:
                        tl_stats[m["key"]].append(sust.get(m["key"], 0))
                except Exception as e:
                    print(f"Error loading TL fraction data: {e}")
    else:
        # Root Directory Parsing (e.g., standard single run experiments)
        root_tl_file = os.path.join(current_exp_dir, f"{CONDITION}_tl.npy")
        if os.path.exists(root_tl_file):
            print("  -> Found single TL baseline file in root directory. Plotting static reference...")
            try:
                data = np.load(root_tl_file, allow_pickle=True).item()
                sust = data.get("sustainability_stats", {})
                # Assign it to a fixed baseline strategy dictionary for flatline rendering
                baselines["TL"] = {}
                for m in metrics_to_plot:
                    baselines["TL"][m["key"]] = sust.get(m["key"], 0)
            except Exception as e:
                print(f"Error loading root TL file: {e}")
        else:
            print("  -> Warning: No Transfer Learning metrics found for this experiment configuration.")

    # Guard check: skip plotting empty contexts
    if not any(baselines["SD"].values()) and not any(baselines["SI"].values()) and not tl_x and "TL" not in baselines:
        print(f"Skipping plot for {experiment}: No valid parameters harvested.")
        continue

    # 3. Generate individual Grid Subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axs = axs.ravel()

    for idx, m in enumerate(metrics_to_plot):
        ax = axs[idx]
        key = m["key"]
        
        # Plot SD Baseline
        if key in baselines["SD"]:
            ax.axhline(y=baselines["SD"][key], color=PARADIGM_STYLES["SD"]["color"], 
                       linestyle=PARADIGM_STYLES["SD"]["linestyle"], linewidth=2, 
                       label=PARADIGM_STYLES["SD"]["label"])
                       
        # Plot SI Baseline
        if key in baselines["SI"]:
            ax.axhline(y=baselines["SI"][key], color=PARADIGM_STYLES["SI"]["color"], 
                       linestyle=PARADIGM_STYLES["SI"]["linestyle"], linewidth=2, 
                       label=PARADIGM_STYLES["SI"]["label"])
                       
        # Scenario A: Plot Transfer Learning as a curve line
        if tl_x:
            ax.plot(tl_x, np.array(tl_stats[key]), marker=PARADIGM_STYLES["TL"]["marker"], 
                    linestyle=PARADIGM_STYLES["TL"]["linestyle"], color=PARADIGM_STYLES["TL"]["color"], 
                    linewidth=2.5, label=PARADIGM_STYLES["TL"]["label"])
                    
        # Scenario B: Plot Transfer Learning as a fixed flat baseline line
        elif "TL" in baselines and key in baselines["TL"]:
            ax.axhline(y=baselines["TL"][key], color=PARADIGM_STYLES["TL"]["color"], 
                       linestyle=PARADIGM_STYLES["TL"]["linestyle"], linewidth=2, 
                       label=PARADIGM_STYLES["TL"]["label"])

        ax.set_ylabel(m["ylabel"], fontweight="bold")
        ax.set_title(m["title"], fontweight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for i in [2, 3]:
        axs[i].set_xlabel("Fine-Tuning Data Fraction (%)", fontweight="bold", labelpad=10)
        axs[i].set_xticks(FT_PERCENTS)

    axs[0].legend(loc="upper left", frameon=True, facecolor="white")

    plt.suptitle(f"Paradigm Sustainability Comparison: {MODEL.upper()} ({CONDITION})\n({experiment})", 
                 fontweight="bold", fontsize=14, y=0.96)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_path = os.path.join(current_exp_dir, f"{CONDITION}_paradigm_sustainability.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  -> Successfully saved plot to: {save_path}")
