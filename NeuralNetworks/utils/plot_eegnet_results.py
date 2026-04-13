import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_1samp

RESULTS_DIR = "../results"
CONDITION = "BSL"  # BLS or "DELAY" or "SENSORY"
MODEL = "eegnet"   # "eegnet", "tcn", "cfc"
CHANCE = 1/3 * 100

CLASS_LABELS = ["Visual", "Spatial", "Verbal"]

# Load results
sd = np.load(os.path.join(RESULTS_DIR, f"{CONDITION}_subject_dependent.npy"), allow_pickle=True).item()
si = np.load(os.path.join(RESULTS_DIR, f"{CONDITION}_subject_independent.npy"), allow_pickle=True).item()

# Extract accuracies
sd_acc = [sd[k]["accuracy"] * 100 for k in sd if k != "group_stats"]
si_acc = [si[k]["accuracy"] * 100 for k in si if k != "group_stats"]

# =========================
# VIOLIN PLOT
# =========================

data = [sd_acc, si_acc]
labels = ["Subject-dependent", "Subject-independent"]

plt.figure(figsize=(7,6))

sns.violinplot(data=data, inner="box", cut=0)
sns.stripplot(data=data, color="black", size=4, jitter=0.2, alpha=0.5)

plt.xticks([0,1], labels)
plt.ylabel("Accuracy (%)")
plt.title(f"EEGNet Performance ({CONDITION})")

plt.axhline(CHANCE, linestyle="--", color="black")

# Stats
for i, accs in enumerate(data):
    t, p = ttest_1samp(accs, CHANCE)
    mean = np.mean(accs)

    if p < 0.001: star = "***"
    elif p < 0.01: star = "**"
    elif p < 0.05: star = "*"
    else: star = "n.s."

    plt.text(i, max(accs)+2, f"{star} {mean:.1f}%", ha="center")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f"{CONDITION}_violin.pdf"))
plt.close()

# =========================
# CONFUSION MATRICES
# =========================

def average_cm(results):
    cms = [results[k]["confusion_matrix"] for k in results if k != "group_stats"]
    return np.mean(cms, axis=0) * 100

cm_sd = average_cm(sd)
cm_si = average_cm(si)

fig, axes = plt.subplots(1,2, figsize=(10,4))

sns.heatmap(cm_sd, annot=True, fmt=".1f", cmap="jet",
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            ax=axes[0])

axes[0].set_title("Subject-dependent")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

sns.heatmap(cm_si, annot=True, fmt=".1f", cmap="jet",
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            ax=axes[1])

axes[1].set_title("Subject-independent")
axes[1].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f"{CONDITION}_confusion.pdf"))
plt.close()