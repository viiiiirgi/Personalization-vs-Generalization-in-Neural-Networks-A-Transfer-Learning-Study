import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.stats import ttest_1samp
from scipy.stats import ttest_rel

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_BASE = os.path.join(BASE_DIR, "results", "eegnet")

# Settings
CHANCE = 33.33
CONDITIONS = ["BSL", "SENSORY", "DELAY"]
CLASS_LABELS = ["Visual", "Spatial", "Verbal"]

STRATEGY_MAP = {
    "si": "Subject-Independent",
    "sd": "Subject-Dependent",
    "tl": "Transfer Learning"
}

mpl.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

def load_condition_results(results_dir, condition, suffix):
    path = os.path.join(results_dir, f"{condition}_{suffix}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing results file: {path}")
    return np.load(path, allow_pickle=True).item()

def subject_keys(condition_results):
    return [k for k in condition_results.keys() if k != "group_stats"]

def extract_accuracy_percent(condition_results):
    keys = subject_keys(condition_results)
    return {k: condition_results[k]["accuracy"] * 100 for k in keys}

def load_results(experiment, suffix, condition):
    path = os.path.join(BASE_DIR, "results", "eegnet", experiment, f"{condition}_{suffix}.npy")
    return np.load(path, allow_pickle=True).item()

def get_subject_accuracy_vector(results_dict):
    keys = sorted([k for k in results_dict.keys() if k != "group_stats"])
    return np.array([results_dict[k]["accuracy"] * 100 for k in keys])

def extract_confusions(condition_results):
    keys = subject_keys(condition_results)
    return [np.array(condition_results[k]["confusion_matrix"]) for k in keys]

def extract_subject_id(filename):
    return filename.split("_", 1)[1] if "_" in filename else filename

def significance_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

def save_latex_table(df, output_dir, filename, caption, label):
    os.makedirs(output_dir, exist_ok=True)
    latex = df.to_latex(
        index=False,
        float_format="%.2f",
        caption=caption,
        label=label,
        escape=False
    )
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(latex)

def clean_sub_id(k):
    base = os.path.basename(k).replace(".npy", "")
    parts = base.split("_")
    for part in parts:
        if "sub-" in part:
            return part
    return parts[-1]

def main(results_dir, suffix):
    strategy_name = STRATEGY_MAP.get(suffix.lower(), suffix.upper())
    print(f"Processing EEGNet {strategy_name} matrix charts inside: {results_dir}")
    
    os.makedirs(results_dir, exist_ok=True)
    results = {cond: load_condition_results(results_dir, cond, suffix) for cond in CONDITIONS}

    # Build accuracy dataframe
    rows = []
    for cond in CONDITIONS:
        acc_dict = extract_accuracy_percent(results[cond])
        for sub, acc in acc_dict.items():
            rows.append({"Condition": cond, "Subject": sub, "Accuracy": acc})
    df = pd.DataFrame(rows)

    # Figure A: Violin plot across BSL/SENSORY/DELAY 
    print(f"Generating {suffix.upper()} violin plot...")
    palette = {"BSL": "lightgray", "SENSORY": "#9BDDB7", "DELAY": "#D5A6E6"}
    fig, ax = plt.subplots(figsize=(8, 6))

    vp = sns.violinplot(
        x="Condition", y="Accuracy", data=df, palette=palette,
        inner=None, cut=0, linewidth=1.2, width=0.6, ax=ax
    )

    for pc in vp.collections:
        pc.set_alpha(0.3)

    sns.stripplot(
        x="Condition", y="Accuracy", data=df, color="black",
        size=4, jitter=0.2, alpha=0.4, zorder=2, ax=ax
    )

    ax.axhline(CHANCE, linestyle="-.", color="black", linewidth=1.5, zorder=0)

    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Mean accuracy'),
        Line2D([0], [0], marker='o', color='white', markeredgecolor='black', markersize=8, linestyle='None', label='Median accuracy'),
        Line2D([0], [0], color='black', lw=6, alpha=0.7, label='Standard deviation'),
        Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=5, alpha=0.4, label='Individual data points'),
        Line2D([0], [0], color='black', linestyle='-.', lw=1.5, label='Chance level')
    ]

    for i, cond in enumerate(CONDITIONS):
        accs = df[df["Condition"] == cond]["Accuracy"].to_numpy()
        mu = np.mean(accs)
        median = np.median(accs)
        std_dev = np.std(accs)

        _, p = ttest_1samp(accs, CHANCE)
        star = significance_star(p)

        ax.hlines(y=mu, xmin=i - 0.25, xmax=i + 0.25, color=palette[cond], linewidth=2.5, zorder=2)
        ax.vlines(x=i, ymin=mu - std_dev, ymax=mu + std_dev, color="#333333", linewidth=6, zorder=3)
        ax.plot(i, median, marker="o", markerfacecolor="white", markeredgecolor="black", markersize=9, zorder=4)
        ax.text(i, np.max(accs) + 4, f"{mu:.0f}% {star}", ha="center", fontsize=11, fontweight="bold", color=palette[cond])

    ax.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=10)
    ax.set_title(f"EEGNet {strategy_name}: Classification accuracy", fontweight="bold", pad=25)
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{suffix}_Figure2_Accuracy.pdf"), bbox_inches="tight")
    plt.close()

    # VARIABILITY PLOT
    print(f"Generating {suffix.upper()} run variability plot...")
    rows_runs = []
    for cond in CONDITIONS:
        keys = subject_keys(results[cond])
        for sub in keys:
            runs = results[cond][sub]["run_accuracies"]
            for r in runs:
                rows_runs.append({"Condition": cond, "Subject": sub, "Accuracy": r * 100})

    df_runs = pd.DataFrame(rows_runs)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(x="Condition", y="Accuracy", data=df_runs, palette=palette, width=0.5, boxprops={'alpha':0.4}, showcaps=True, ax=ax)
    sns.stripplot(x="Condition", y="Accuracy", data=df_runs, color="black", alpha=0.25, jitter=0.2, ax=ax)
    ax.axhline(CHANCE, linestyle="--", color="gray", alpha=0.7, label="Chance Level (33.3%)")
    
    variability_legends = [
        Line2D([0], [0], color="black", lw=2, label="Median"),
        Line2D([0], [0], color="black", lw=6, alpha=0.3, label="Interquartile range (25–75%)"),
        Line2D([0], [0], color="black", lw=1, label="Whiskers (±1.5 IQR)"),
        Line2D([0], [0], marker='o', linestyle='None', markersize=5, alpha=0.3, label="Single runs"),
        Line2D([0], [0], color="gray", lw=1.5, linestyle="-.", label="Chance level")
    ]
    ax.legend(handles=variability_legends, loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    ax.set_title(f"Run-level variability within subjects ({strategy_name})", fontweight="bold", pad=15)
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_xlabel("")
    plt.savefig(os.path.join(results_dir, f"{suffix}_RunVariability.pdf"), bbox_inches="tight")
    plt.close()
    
    # PER CLASS VARIABILITY PLOT
    print(f"Generating {suffix.upper()} per-class accuracy plot...")
    rows_class = []
    for cond in CONDITIONS:
        keys = subject_keys(results[cond])
        for sub in keys:
            accs = results[cond][sub]["per_class_accuracy"]
            for i, cls in enumerate(CLASS_LABELS):
                rows_class.append({"Condition": cond, "Class": cls, "Accuracy": accs[i] * 100})

    df_class = pd.DataFrame(rows_class)
    plt.figure(figsize=(8,6))

    class_palette = {"Visual": "#5D9CEC", "Spatial": "#FC6E51", "Verbal": "#A0D468"}
    sns.pointplot(x="Condition", y="Accuracy", hue="Class", data=df_class, palette=class_palette, errorbar="se", dodge=0.3, markers="o", linestyles="-.")
    
    legend_elements_class = [
        Line2D([0], [0], color="#5D9CEC", lw=6, label="Visual"),
        Line2D([0], [0], color="#FC6E51", lw=6, label="Spatial"),
        Line2D([0], [0], color="#A0D468", lw=6, label="Verbal"),
        Line2D([0], [0], color="black", lw=1.5, label="Mean accuracy"),
        Line2D([0], [0], color="black", lw=1, linestyle="-.", label="Standard error (SEM)")
    ]
    plt.legend(handles=legend_elements_class, title="Class & statistics", frameon=False, loc="upper right")
    plt.axhline(CHANCE, linestyle="-.", color="black", linewidth=1.5)
    plt.title(f"Per-Class Classification Performance ({strategy_name})", fontweight="bold", pad=15)
    plt.ylabel("Mean Accuracy (%)")
    plt.xlabel("")
    plt.savefig(os.path.join(results_dir, f"{suffix}_PerClassAccuracy.pdf"), bbox_inches="tight")
    plt.close()

    # Figure B: Individual paired SENSORY vs DELAY
    print(f"Generating {suffix.upper()} paired sensory-delay plot...")
    fig, ax = plt.subplots(figsize=(7, 6))

    sens = {extract_subject_id(k): v for k, v in extract_accuracy_percent(results["SENSORY"]).items()}
    dly  = {extract_subject_id(k): v for k, v in extract_accuracy_percent(results["DELAY"]).items()}
    common_subjects = sorted(set(sens.keys()).intersection(set(dly.keys())))
    paired_data = [(sub, sens[sub], dly[sub]) for sub in common_subjects]
    paired_data.sort(key=lambda x: x[1])

    cmap = mpl.colormaps["jet"].resampled(max(1, len(paired_data)))
    legend_dots = []
    for rank, (sub, s_acc, d_acc) in enumerate(paired_data):
        color = cmap(rank)
        ax.plot([0, 1], [s_acc, d_acc], color=color, alpha=0.7, marker="o", markersize=6, zorder=1)
        legend_dots.append(Line2D([0], [0], color=color, marker="o", linestyle="None", markersize=6, label=f"{rank + 1}"))

    ax.axhline(CHANCE, linestyle="-.", color="black", linewidth=1.5, zorder=0)
    ax.legend(handles=legend_dots, title="Accuracy\nat Sensory", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, handletextpad=0.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sensory", "Delay"], fontsize=12)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(10, 100)
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_title(f"EEGNet {suffix.upper()}: Classification accuracy per individual", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{suffix}_Figure6_IndividualAccuracy.pdf"), bbox_inches="tight")
    plt.close()

    # Figure C: Confusion matrices
    print(f"Generating {suffix.upper()} confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, cond in enumerate(CONDITIONS):
        cms = extract_confusions(results[cond])
        mean_cm = np.mean(cms, axis=0) * 100
        sns.heatmap(
            mean_cm, annot=True, fmt=".2f", cmap="jet", vmin=0, vmax=100,
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, cbar=(i == 2),
            cbar_kws={"label": "Classification accuracy(%)"} if i == 2 else None, ax=axes[i]
        )
        axes[i].set_title(cond, fontweight="bold")
        axes[i].set_xlabel("Tested class")
        if i == 0: axes[i].set_ylabel("Trained class")

    plt.suptitle(f"EEGNet {strategy_name}: Confusion matrices", fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{suffix}_Figure5_ConfusionMatrices.pdf"), bbox_inches="tight")
    plt.close()

    # Stats printout 
    print(f"\n===== EEGNet {strategy_name.upper()} T-TEST VS CHANCE =====")
    for cond in CONDITIONS:
        accs = list(extract_accuracy_percent(results[cond]).values())
        t, p = ttest_1samp(accs, CHANCE)
        print(f"[{cond}] Mean accuracy: {np.mean(accs):.2f}% | t={t:.2f} | p={p:.4e}")

    # PAIRED T-TESTS BETWEEN METHODS
    print("\n===== PAIRED T-TESTS BETWEEN METHODS =====")
    comparisons = [("TL", "tl", "SI", "si"), ("TL", "tl", "SD", "sd")]
    current_experiment_name = os.path.basename(results_dir)

    for cond in CONDITIONS:
        print(f"### CONDITION: {cond}")
        for name_a, suffix_a, name_b, suffix_b in comparisons:
            try:
                res_a = load_results(current_experiment_name, suffix_a, cond)
                res_b = load_results(current_experiment_name, suffix_b, cond)
                vec_a = get_subject_accuracy_vector(res_a)
                vec_b = get_subject_accuracy_vector(res_b)
                t, p = ttest_rel(vec_a, vec_b)
                print(f"  {name_a} vs {name_b}: t={t:.3f}, p={p:.4e}")
            except FileNotFoundError:
                print(f"  {name_a} vs {name_b}: Comparative evaluation arrays missing. Skipping.")

    # PSEUDO VS NO-PSEUDO
    print("\n===== PSEUDO VS NO-PSEUDO =====")
    for cond in CONDITIONS:
        try:
            pseudo_res = load_results("AllData_PseudoTrainAndTest", suffix, cond)
            nopseudo_res = load_results("AllData_NoPseudo", suffix, cond)
            pseudo_vec = get_subject_accuracy_vector(pseudo_res)
            nopseudo_vec = get_subject_accuracy_vector(nopseudo_res)
            t, p = ttest_rel(pseudo_vec, nopseudo_vec)
            print(f"### CONDITION: {cond} | Pseudo vs NoPseudo ({suffix.upper()}): t={t:.3f}, p={p:.4e}")
        except FileNotFoundError:
            print(f"### CONDITION: {cond} | Pseudo comparative files missing. Skipping comparison.")

    # MAIN PERFORMANCE TABLE
    table_rows = []
    for cond in CONDITIONS:
        accs = list(extract_accuracy_percent(results[cond]).values())
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        ci95 = 1.96 * (std_acc / np.sqrt(len(accs)))
        t, p = ttest_1samp(accs, CHANCE)
        table_rows.append({
            "Condition": cond, "Mean Accuracy (%)": mean_acc,
            "Std (%)": std_acc, "95% CI": ci95, "t-value": t, "p-value": p
        })

    df_stats = pd.DataFrame(table_rows)
    save_latex_table(df_stats, results_dir, f"{suffix}_MainStats.tex", f"EEGNet {strategy_name} performance.", f"tab:{suffix}_main_stats")

    # PER-CLASS TABLE
    class_rows = []
    for cond in CONDITIONS:
        per_class_all = [results[cond][sub]["per_class_accuracy"] * 100 for sub in subject_keys(results[cond])]
        per_class_mean = np.mean(per_class_all, axis=0)
        class_rows.append({
            "Condition": cond, "Visual": per_class_mean[0], "Spatial": per_class_mean[1], "Verbal": per_class_mean[2]
        })

    df_class_table = pd.DataFrame(class_rows)
    save_latex_table(df_class_table, results_dir, f"{suffix}_PerClassStats.tex", f"Per-class accuracy for EEGNet {strategy_name}.", f"tab:{suffix}_perclass")

    # SUBJECT TABLE
    subject_rows = []
    subjects = sorted(list(set([clean_sub_id(k) for k in subject_keys(results[CONDITIONS[0]])])))

    for sub in subjects:
        row = {"Subject": sub}
        for cond in CONDITIONS:
            cond_dict = extract_accuracy_percent(results[cond])
            matching_key = next((k for k in cond_dict.keys() if clean_sub_id(k) == sub), None)
            row[cond] = cond_dict[matching_key] if matching_key is not None else np.nan
        subject_rows.append(row)

    df_subjects = pd.DataFrame(subject_rows)
    save_latex_table(df_subjects, results_dir, f"{suffix}_SubjectAccuracies.tex", f"Subject decoding accuracies ({strategy_name}).", f"tab:{suffix}_subjects")
    print(f"All figures and LaTeX tables written cleanly into: {results_dir}\n")

if __name__ == "__main__":
    if not os.path.exists(RESULTS_BASE):
        print(f"Error: Base directory '{RESULTS_BASE}' missing. Run evaluation script first.")
    else:
        suffixes = ["sd", "si", "tl"]

        # Recursively walk through ALL folders inside RESULTS_BASE (handles FT_10, FT_25, etc. safely)
        for root, dirs, files in os.walk(RESULTS_BASE):
            for suffix in suffixes:
                # Check if this target directory contains a valid experiment profile (e.g. BSL_tl.npy)
                sample_file = os.path.join(root, f"BSL_{suffix}.npy")
                if os.path.exists(sample_file):
                    print(f"\n========================================================")
                    print(f"Found active result folder: {root}")
                    print(f"Running pipeline for strategy suffix: [{suffix.upper()}]")
                    print(f"========================================================")
                    
                    try:
                        main(root, suffix)
                    except Exception as e:
                        print(f"Skipping visualization for {suffix.upper()} in {root} due to error: {e}")