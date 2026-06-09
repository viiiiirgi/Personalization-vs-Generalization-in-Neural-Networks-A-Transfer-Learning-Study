import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL = "eegnet"
EXPERIMENT = "AllData_NoPseudo"
CONDITION = "SENSORY" #BSL or SENSORY OR DELAY

FT_PERCENTS = [10, 25, 50, 75, 100]

means = []
cis = []
x_labels = []

for p in FT_PERCENTS:

    fraction = p / 100.0
    absolute_dataset_percent = int(round(fraction * 70.0))
    x_labels.append(absolute_dataset_percent)

    file = os.path.join(
        RESULTS_DIR,
        MODEL,
        EXPERIMENT,
        f"FT_{absolute_dataset_percent}",
        f"{CONDITION}_tl.npy"
    )

    data = np.load(file, allow_pickle=True).item()

    stats = data["group_stats"]

    means.append(stats["mean_accuracy"])
    cis.append(stats["ci95"])

plt.figure(figsize=(7,5))

x = np.array(FT_PERCENTS)

means = np.array(means)
cis = np.array(cis)

plt.plot(x, means, marker="o")

plt.fill_between(
    x,
    means - cis,
    means + cis,
    alpha=0.3
)

plt.xlabel("Fine-tuning data (%)")
plt.ylabel("Accuracy")
plt.title(f"{MODEL} - {CONDITION}")

plt.grid(True)

save_path = os.path.join(
    RESULTS_DIR,
    MODEL,
    EXPERIMENT,
    f"{CONDITION}_finetune_curve.png"
)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", save_path)