import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_1samp
import matplotlib as mpl
from matplotlib.lines import Line2D

OUTPUT_DIR = "results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# SETTINGS
CHANCE = 33.33
CONDITIONS = ["BSL", "SENSORY", "DELAY"]
CLASS_LABELS = ["Visual", "Spatial", "Verbal"]

mpl.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300
})


# LOAD RESULTS
INPUT_PATH = os.path.join(OUTPUT_DIR, "turoman_results.pkl")
print("Loading results from turoman_results.pkl...")
with open(INPUT_PATH, "rb") as f:
    results = pickle.load(f)

# Create DataFrame for Violin Plots
rows = []
for cond in CONDITIONS:
    for sub in results[cond]:
        rows.append({
            "Condition": cond,
            "Subject": sub,
            "Accuracy": results[cond][sub]["accuracy"] * 100
        })
df = pd.DataFrame(rows)

# ============================
# FIGURE 1: VIOLIN PLOTS
print("Generating Figure 1: Time-Averaged Accuracy Violin Plots...")
palette = {
    "BSL": "lightgray",
    "SENSORY": "#9BDDB7",  
    "DELAY": "#D5A6E6"     
}

fig, ax = plt.subplots(figsize=(8, 6))

sns.violinplot(x="Condition", y="Accuracy", data=df, palette=palette, 
               inner="box", cut=0, linewidth=1.2, width=0.6, ax=ax, alpha=0.3)

sns.stripplot(x="Condition", y="Accuracy", data=df, color="black", 
              size=4, jitter=0.2, alpha=0.4, zorder=2, ax=ax)

ax.axhline(CHANCE, linestyle="--", color="black", linewidth=1.5, zorder=0)

legend_elements = []

for i, cond in enumerate(CONDITIONS):
    accs = df[df["Condition"]==cond]["Accuracy"]
    mu = accs.mean()
    t, p = ttest_1samp(accs, CHANCE)
    sd = accs.std()
    md = accs.median()
    
    ax.hlines(y=mu, xmin=i-0.2, xmax=i+0.2, color=palette[cond], linewidth=4, zorder=3)

    if p < 0.001: star = "***"
    elif p < 0.01: star = "**"
    elif p < 0.05: star = "*"
    else: star = "n.s."
    
    y_pos = accs.max() + 5
    ax.text(i, y_pos, f"{star} {mu:.1f}%", 
            ha="center", fontsize=12, fontweight='bold', color=palette[cond])

    legend_elements.append(Line2D([0], [0], color=palette[cond], lw=4, label=f"{cond} Mean"))

legend_elements.append(Line2D([0], [0], color='black', marker='o', linestyle='None', 
                              markersize=5, alpha=0.6, label='Individual data points'))
legend_elements.append(Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label=f'Chance ({CHANCE}%)'))

ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=10)

ax.set_title("Classification of time-averaged patterns", fontweight='bold', pad=25)
ax.set_ylabel("Classification accuracy (%)")
ax.set_xlabel("")
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure2_TimeAveragedAccuracy.pdf"), bbox_inches='tight')
plt.close()

# FIGURE 2: INDIVIDUAL SENSORY VS DELAY
print("Generating Figure 6: Individual Sensory vs Delay paired plot...")
fig, ax = plt.subplots(figsize=(7, 6))

subjects = list(results["SENSORY"].keys())
sensory_acc = [results["SENSORY"][sub]["accuracy"] * 100 for sub in subjects]
delay_acc = [results["DELAY"][sub]["accuracy"] * 100 for sub in subjects]

paired_data = list(zip(subjects, sensory_acc, delay_acc))
paired_data.sort(key=lambda x: x[1]) 

cmap = plt.cm.get_cmap('jet', len(paired_data))

legend_dots = []

# Plot individual subject trajectories with assigned colors
for rank, (sub, s_acc, d_acc) in enumerate(paired_data):
    color = cmap(rank)
    ax.plot([1, 2], [s_acc, d_acc], color=color, alpha=0.7, marker="o", markersize=6, zorder=1)
    
    legend_dots.append(Line2D([0], [0], color=color, marker='o', linestyle='None', markersize=6, label=f"{rank+1}"))

# Chance level line
ax.axhline(CHANCE, linestyle="--", color="black", linewidth=1.5, zorder=0)

ax.legend(handles=legend_dots, title="Accuracy\nat Sensory", bbox_to_anchor=(1.05, 1), 
          loc='upper left', frameon=False, handletextpad=0.1)

ax.set_xticks([1, 2])
ax.set_xticklabels(["Sensory", "Delay"], fontsize=12)
ax.set_xlim(0.8, 2.2)
ax.set_ylim(10, 100)
ax.set_ylabel("Classification accuracy (%)")
ax.set_title("Classification accuracy per individual", fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure6_IndividualAccuracy.pdf"), bbox_inches='tight')
plt.close()

# FIGURE 3: CONFUSION MATRICES (All Subjects)
print("Generating Figure 5: Confusion Matrices for All Subjects...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, cond in enumerate(CONDITIONS):
    cms = [results[cond][sub]["confusion"] for sub in results[cond]]
    mean_cm = np.mean(cms, axis=0) * 100

    sns.heatmap(mean_cm, annot=True, fmt=".2f", cmap="jet", vmin=10, vmax=80,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, 
                cbar=(i==2), cbar_kws={'label': 'Classification accuracy(%)'} if i==2 else None, ax=axes[i])

    axes[i].set_title(cond, fontweight='bold')
    axes[i].set_xlabel("Tested class")
    if i == 0:
        axes[i].set_ylabel("Trained class")

plt.suptitle("Confusion matrices", fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Figure5_ConfusionMatrices.pdf"), bbox_inches='tight')
plt.close()


# STATISTICS PRINTOUT
print("\n===== T-TEST VS CHANCE =====")
for cond in CONDITIONS:
    accs = [results[cond][sub]["accuracy"] * 100 for sub in results[cond]]
    t, p = ttest_1samp(accs, CHANCE)
    
    print(f"\n[{cond}]")
    print(f"Mean accuracy: {np.mean(accs):.2f}%")
    print(f"t-value: {t:.2f}")
    print(f"p-value: {p:.4e}")

print("\nAll figures saved successfully as PDFs (300 dpi) in the current directory.")
