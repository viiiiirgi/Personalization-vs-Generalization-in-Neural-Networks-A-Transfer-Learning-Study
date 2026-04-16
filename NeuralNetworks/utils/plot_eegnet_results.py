import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.stats import ttest_1samp


# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "eegnet")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "eegnet", "plot_like_turoman")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Settings
CHANCE = 33.33
CONDITIONS = ["BSL", "SENSORY", "DELAY"]
CLASS_LABELS = ["Visual", "Spatial", "Verbal"]
SUFFIX = "si"  # subject-independent files: {COND}_si.npy

mpl.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def load_condition_results(condition):
    path = os.path.join(RESULTS_DIR, f"{condition}_{SUFFIX}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing results file: {path}")
    return np.load(path, allow_pickle=True).item()


def subject_keys(condition_results):
    return [k for k in condition_results.keys() if k != "group_stats"]


def extract_accuracy_percent(condition_results):
    keys = subject_keys(condition_results)
    return {k: condition_results[k]["accuracy"] * 100 for k in keys}


def extract_confusions(condition_results):
    keys = subject_keys(condition_results)
    return [np.array(condition_results[k]["confusion_matrix"]) for k in keys]


def significance_star(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    print("Loading EEGNet SI results...")
    results = {cond: load_condition_results(cond) for cond in CONDITIONS}

    # Build accuracy dataframe (Turoman figure style)
    rows = []
    for cond in CONDITIONS:
        acc_dict = extract_accuracy_percent(results[cond])
        for sub, acc in acc_dict.items():
            rows.append({"Condition": cond, "Subject": sub, "Accuracy": acc})
    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Figure A: Violin plot across BSL/SENSORY/DELAY (Turoman style)
    # ------------------------------------------------------------------
    print("Generating SI violin plot...")
    palette = {
        "BSL": "lightgray",
        "SENSORY": "#9BDDB7",
        "DELAY": "#D5A6E6",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        x="Condition",
        y="Accuracy",
        data=df,
        palette=palette,
        inner="box",
        cut=0,
        linewidth=1.2,
        width=0.6,
        ax=ax,
        alpha=0.3,
    )
    sns.stripplot(
        x="Condition",
        y="Accuracy",
        data=df,
        color="black",
        size=4,
        jitter=0.2,
        alpha=0.4,
        zorder=2,
        ax=ax,
    )

    ax.axhline(CHANCE, linestyle="--", color="black", linewidth=1.5, zorder=0)

    legend_elements = []
    for i, cond in enumerate(CONDITIONS):
        accs = df[df["Condition"] == cond]["Accuracy"].to_numpy()
        mu = np.mean(accs)
        _, p = ttest_1samp(accs, CHANCE)
        star = significance_star(p)

        ax.hlines(y=mu, xmin=i - 0.2, xmax=i + 0.2, color=palette[cond], linewidth=4, zorder=3)
        ax.text(i, np.max(accs) + 5, f"{star} {mu:.1f}%", ha="center", fontsize=12, fontweight="bold", color=palette[cond])
        legend_elements.append(Line2D([0], [0], color=palette[cond], lw=4, label=f"{cond} Mean"))

    legend_elements.append(Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=5, alpha=0.6, label="Individual data points"))
    legend_elements.append(Line2D([0], [0], color="black", linewidth=1.5, linestyle="--", label=f"Chance ({CHANCE}%)"))
    ax.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=10)

    ax.set_title("EEGNet Subject-Independent: Classification accuracy", fontweight="bold", pad=25)
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "SI_Figure2_Accuracy.pdf"), bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Figure B: Individual paired SENSORY vs DELAY (Turoman style)
    # ------------------------------------------------------------------
    print("Generating SI paired sensory-delay plot...")
    fig, ax = plt.subplots(figsize=(7, 6))

    sens = extract_accuracy_percent(results["SENSORY"])
    dly = extract_accuracy_percent(results["DELAY"])
    common_subjects = sorted(set(sens.keys()).intersection(set(dly.keys())))
    paired_data = [(sub, sens[sub], dly[sub]) for sub in common_subjects]
    paired_data.sort(key=lambda x: x[1])  # same ordering logic as Turoman

    cmap = plt.cm.get_cmap("jet", max(1, len(paired_data)))
    legend_dots = []
    for rank, (sub, s_acc, d_acc) in enumerate(paired_data):
        color = cmap(rank)
        ax.plot([1, 2], [s_acc, d_acc], color=color, alpha=0.7, marker="o", markersize=6, zorder=1)
        legend_dots.append(Line2D([0], [0], color=color, marker="o", linestyle="None", markersize=6, label=f"{rank + 1}"))

    ax.axhline(CHANCE, linestyle="--", color="black", linewidth=1.5, zorder=0)
    ax.legend(
        handles=legend_dots,
        title="Accuracy\nat Sensory",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False,
        handletextpad=0.1,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Sensory", "Delay"], fontsize=12)
    ax.set_xlim(0.8, 2.2)
    ax.set_ylim(10, 100)
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_title("EEGNet SI: Classification accuracy per individual", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "SI_Figure6_IndividualAccuracy.pdf"), bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Figure C: Confusion matrices for BSL/SENSORY/DELAY (Turoman style)
    # ------------------------------------------------------------------
    print("Generating SI confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, cond in enumerate(CONDITIONS):
        cms = extract_confusions(results[cond])
        mean_cm = np.mean(cms, axis=0) * 100
        sns.heatmap(
            mean_cm,
            annot=True,
            fmt=".2f",
            cmap="jet",
            vmin=0,
            vmax=100,
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            cbar=(i == 2),
            cbar_kws={"label": "Classification accuracy(%)"} if i == 2 else None,
            ax=axes[i],
        )
        axes[i].set_title(cond, fontweight="bold")
        axes[i].set_xlabel("Tested class")
        if i == 0:
            axes[i].set_ylabel("Trained class")

    plt.suptitle("EEGNet SI: Confusion matrices", fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "SI_Figure5_ConfusionMatrices.pdf"), bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Stats printout (Turoman style)
    # ------------------------------------------------------------------
    print("\n===== EEGNet SI T-TEST VS CHANCE =====")
    for cond in CONDITIONS:
        accs = list(extract_accuracy_percent(results[cond]).values())
        t, p = ttest_1samp(accs, CHANCE)
        print(f"\n[{cond}]")
        print(f"Mean accuracy: {np.mean(accs):.2f}%")
        print(f"t-value: {t:.2f}")
        print(f"p-value: {p:.4e}")

    print(f"\nAll SI figures saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

