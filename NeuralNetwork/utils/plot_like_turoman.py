import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.stats import ttest_1samp, ttest_rel

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EEGNET_RESULTS_DIR = os.path.join(BASE_DIR, "results", "eegnet")
LDA_RESULTS_PATH = os.path.join(BASE_DIR, "results", "turoman_results.pkl")

EXPERIMENTS = [
    "AllData_NoPseudo",
    "AllData_PseudoTrainAndTest",
    "AllData_PseudoTrainOnly",
    "Downsample_1000_NoPseudo",
    "Downsample_MinSubject_NoPseudo"
]

DOWNSAMPLE_EXPERIMENTS = {
    "Downsample_MinSubject_NoPseudo": "Matched min size",
    "Downsample_1000_NoPseudo": "1000 trials",
    "AllData_NoPseudo": "Full data"
}

CHANCE = 33.33
CONDITIONS = ["BSL", "SENSORY", "DELAY"]
STRATEGIES = ["sd", "si", "tl"]
CLASS_LABELS = ["Visual", "Spatial", "Verbal"]

METHOD_LABELS = {"lda": "LDA", "sd": "SD", "si": "SI", "tl": "TL"}

mpl.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def subject_keys(results_dict):
    return [k for k in results_dict.keys() if k not in ["group_stats", "sustainability_stats"]]


def clean_subject_id(k):
    base = os.path.basename(k).replace(".npy", "")
    parts = base.split("_")
    for part in parts:
        if "sub-" in part:
            return part
    return parts[-1]


def load_eegnet_results(condition, strategy, experiment):
    path = os.path.join(EEGNET_RESULTS_DIR, experiment, f"{condition}_{strategy}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


def get_accuracy_dict(results_dict):
    return {
        clean_subject_id(k): results_dict[k]["accuracy"] * 100
        for k in subject_keys(results_dict)
    }


def paired_vectors(dict_a, dict_b):
    common_subjects = sorted(set(dict_a.keys()).intersection(dict_b.keys()))
    return (
        np.array([dict_a[s] for s in common_subjects]),
        np.array([dict_b[s] for s in common_subjects]),
        common_subjects
    )


def save_latex_table(df, output_dir, filename, caption, label):
    path = os.path.join(output_dir, filename)

    df = df.copy()
    df.columns = [col.replace("%", r"\%") for col in df.columns]

    latex = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        formatters={
            "Mean accuracy (\\%)": lambda x: f"{x:.3f}",
            "Mean EEGNet (\\%)": lambda x: f"{x:.3f}",
            "Mean LDA (\\%)": lambda x: f"{x:.3f}",
            "Mean A (\\%)": lambda x: f"{x:.3f}",
            "Mean B (\\%)": lambda x: f"{x:.3f}",
            "Mean smoothed (\\%)": lambda x: f"{x:.3f}",
            "Mean original (\\%)": lambda x: f"{x:.3f}",
            "SD": lambda x: f"{x:.3f}",
            "CI95": lambda x: f"{x:.3f}",
            "t": lambda x: f"{x:.3f}",
            "df": lambda x: f"{int(x)}",
            "p one-tailed": lambda x: f"{x:.6e}",
            "p two-tailed": lambda x: f"{x:.6e}",
            "N": lambda x: f"{int(x)}"
        }
    )

    with open(path, "w") as f:
        f.write(latex)

    print("Saved:", path)


def plot_old_style_violin(df, group_order, group_labels, group_palette, title, save_path, condition_positions=None):
    fig, ax = plt.subplots(figsize=(15, 6))

    vp = sns.violinplot(
        x="Group",
        y="Accuracy",
        data=df,
        order=group_order,
        palette=group_palette,
        inner=None,
        cut=0,
        linewidth=1.2,
        width=0.65,
        ax=ax
    )

    for pc in vp.collections:
        pc.set_alpha(0.3)

    sns.stripplot(
        x="Group",
        y="Accuracy",
        data=df,
        order=group_order,
        color="black",
        size=3.5,
        jitter=0.18,
        alpha=0.4,
        zorder=2,
        ax=ax
    )

    ax.axhline(CHANCE, linestyle="-.", color="black", linewidth=1.5, zorder=0)

    for i, group in enumerate(group_order):
        accs = df[df["Group"] == group]["Accuracy"].to_numpy()

        if len(accs) == 0:
            continue

        mu = np.mean(accs)
        median = np.median(accs)
        std_dev = np.std(accs, ddof=1)

        color = group_palette[group]

        ax.hlines(mu, i - 0.22, i + 0.22, color=color, linewidth=2.5, zorder=3)
        ax.vlines(i, mu - std_dev, mu + std_dev, color="#333333", linewidth=5, alpha=0.7, zorder=4)
        ax.plot(i, median, marker="o", markerfacecolor="white", markeredgecolor="black", markersize=8, zorder=5)
        ax.text(i, np.max(accs) + 3, f"{mu:.0f}%", ha="center", fontsize=9, fontweight="bold", color=color)

    if condition_positions is not None:
        for x in condition_positions["separators"]:
            ax.axvline(x, color="black", linewidth=0.8, alpha=0.25)

        for label, x in condition_positions["labels"]:
            ax.text(x, 103, label, ha="center", fontsize=13, fontweight="bold")

    legend_elements = [
        Line2D([0], [0], color="black", lw=2, label="Mean accuracy"),
        Line2D([0], [0], marker="o", color="white", markeredgecolor="black", markersize=8, linestyle="None", label="Median accuracy"),
        Line2D([0], [0], color="black", lw=6, alpha=0.7, label="Standard deviation"),
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=5, alpha=0.4, label="Individual data points"),
        Line2D([0], [0], color="black", linestyle="-.", lw=1.5, label="Chance level")
    ]

    ax.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=10)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_labels, rotation=25 if max(len(x) for x in group_labels) > 5 else 0, ha="right" if max(len(x) for x in group_labels) > 5 else "center")
    ax.set_title(title, fontweight="bold", pad=25)
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def make_main_violin_plot(experiment, output_dir, lda_results):
    print(f"Generating aggregate violin plot for {experiment}...")

    rows = []

    for condition in CONDITIONS:
        for sub in subject_keys(lda_results[condition]):
            rows.append({
                "Condition": condition,
                "Method": "LDA",
                "Subject": clean_subject_id(sub),
                "Accuracy": lda_results[condition][sub]["accuracy"] * 100
            })

        for strategy in STRATEGIES:
            res = load_eegnet_results(condition, strategy, experiment)
            if res is None:
                continue

            for sub, acc in get_accuracy_dict(res).items():
                rows.append({
                    "Condition": condition,
                    "Method": METHOD_LABELS[strategy],
                    "Subject": sub,
                    "Accuracy": acc
                })

    df = pd.DataFrame(rows)

    method_palette = {
        "LDA": "lightgray",
        "SD": "#4A90E2",
        "SI": "#E67E22",
        "TL": "#D5A6E6"
    }

    method_order = ["LDA", "SD", "SI", "TL"]

    group_order, group_labels, group_colors = [], [], []

    df["Group"] = df["Condition"].astype(str) + "_" + df["Method"].astype(str)

    for cond in CONDITIONS:
        for method in method_order:
            group = f"{cond}_{method}"
            if group in df["Group"].values:
                group_order.append(group)
                group_labels.append(method)
                group_colors.append(method_palette[method])

    group_palette = dict(zip(group_order, group_colors))

    condition_positions = {
        "separators": [3.5, 7.5],
        "labels": [("Baseline", 1.5), ("Sensory", 5.5), ("Delay", 9.5)]
    }

    plot_old_style_violin(
        df=df,
        group_order=group_order,
        group_labels=group_labels,
        group_palette=group_palette,
        title=f"{experiment}: classification accuracy across methods and task phases",
        save_path=os.path.join(output_dir, "Aggregate_LDA_SD_SI_TL_Accuracy.pdf"),
        condition_positions=condition_positions
    )


def make_individual_accuracy_plots(experiment, output_dir):
    for strategy in STRATEGIES:
        sensory_res = load_eegnet_results("SENSORY", strategy, experiment)
        delay_res = load_eegnet_results("DELAY", strategy, experiment)

        if sensory_res is None or delay_res is None:
            continue

        print(f"Generating individual Sensory vs Delay plot for {experiment} {strategy.upper()}...")

        sensory_acc = get_accuracy_dict(sensory_res)
        delay_acc = get_accuracy_dict(delay_res)

        common_subjects = sorted(set(sensory_acc.keys()).intersection(delay_acc.keys()))
        paired_data = [(sub, sensory_acc[sub], delay_acc[sub]) for sub in common_subjects]
        paired_data.sort(key=lambda x: x[1])

        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = mpl.colormaps["jet"].resampled(max(1, len(paired_data)))
        legend_dots = []

        for rank, (sub, s_acc, d_acc) in enumerate(paired_data):
            color = cmap(rank)
            ax.plot([1, 2], [s_acc, d_acc], color=color, alpha=0.7, marker="o", markersize=6, zorder=1)
            legend_dots.append(Line2D([0], [0], color=color, marker="o", linestyle="None", markersize=6, label=f"{rank + 1}"))

        ax.axhline(CHANCE, linestyle="--", color="black", linewidth=1.5, zorder=0)
        ax.legend(handles=legend_dots, title="Accuracy\nat Sensory", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, handletextpad=0.1)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Sensory", "Delay"])
        ax.set_xlim(0.8, 2.2)
        ax.set_ylim(10, 100)
        ax.set_ylabel("Classification accuracy (%)")
        ax.set_title(f"{experiment} EEGNet {strategy.upper()}: individual accuracy", fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{strategy}_IndividualAccuracy.pdf"), bbox_inches="tight")
        plt.close()


def make_confusion_matrices(experiment, output_dir):
    for strategy in STRATEGIES:
        if load_eegnet_results("BSL", strategy, experiment) is None:
            continue

        print(f"Generating confusion matrices for {experiment} {strategy.upper()}...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for i, condition in enumerate(CONDITIONS):
            res = load_eegnet_results(condition, strategy, experiment)

            if res is None:
                axes[i].axis("off")
                continue

            cms = [np.array(res[sub]["confusion_matrix"]) for sub in subject_keys(res)]
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
                cbar_kws={"label": "Classification accuracy (%)"} if i == 2 else None,
                ax=axes[i]
            )

            axes[i].set_title(condition, fontweight="bold")
            axes[i].set_xlabel("Predicted class")

            if i == 0:
                axes[i].set_ylabel("True class")

        plt.suptitle(f"{experiment} EEGNet {strategy.upper()}: confusion matrices", fontweight="bold", y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{strategy}_ConfusionMatrices.pdf"), bbox_inches="tight")
        plt.close()


def make_rq1_1_table(experiment, output_dir):
    rq1_rows = []

    for strategy in STRATEGIES:
        for condition in CONDITIONS:
            res = load_eegnet_results(condition, strategy, experiment)
            if res is None:
                continue

            accs = np.array(list(get_accuracy_dict(res).values()))
            t, p = ttest_1samp(accs, CHANCE, alternative="greater")

            mean_acc = np.mean(accs)
            sd_acc = np.std(accs, ddof=1)
            n = len(accs)
            ci95 = 1.96 * sd_acc / np.sqrt(n)

            rq1_rows.append({
                "Strategy": METHOD_LABELS[strategy],
                "Condition": condition,
                "Mean accuracy (%)": mean_acc,
                "SD": sd_acc,
                "CI95": ci95,
                "t": t,
                "df": n - 1,
                "p one-tailed": p,
                "N": n
            })

    if len(rq1_rows) == 0:
        return

    save_latex_table(
        pd.DataFrame(rq1_rows),
        output_dir,
        "RQ1_1_EEGNet_vs_chance.tex",
        f"{experiment}: one-sample t-tests against chance level for EEGNet classification accuracy.",
        f"tab:{experiment.lower()}_rq1_1_eegnet_vs_chance"
    )


def make_rq1_2_table(experiment, output_dir, lda_results):
    rq2_rows = []

    for strategy in STRATEGIES:
        for condition in CONDITIONS:
            eegnet_res = load_eegnet_results(condition, strategy, experiment)
            if eegnet_res is None:
                continue

            eegnet_acc = get_accuracy_dict(eegnet_res)
            lda_acc = get_accuracy_dict(lda_results[condition])
            vec_eegnet, vec_lda, subjects = paired_vectors(eegnet_acc, lda_acc)

            if len(subjects) < 2:
                continue

            t, p = ttest_rel(vec_eegnet, vec_lda, alternative="greater")

            rq2_rows.append({
                "Comparison": f"{METHOD_LABELS[strategy]} vs LDA",
                "Condition": condition,
                "Mean EEGNet (%)": np.mean(vec_eegnet),
                "Mean LDA (%)": np.mean(vec_lda),
                "t": t,
                "df": len(subjects) - 1,
                "p one-tailed": p,
                "N": len(subjects)
            })

    if len(rq2_rows) == 0:
        return

    save_latex_table(
        pd.DataFrame(rq2_rows),
        output_dir,
        "RQ1_2_EEGNet_vs_LDA.tex",
        f"{experiment}: paired t-tests comparing EEGNet classification accuracy against the LDA baseline.",
        f"tab:{experiment.lower()}_rq1_2_eegnet_vs_lda"
    )


def make_rq1_3_table(experiment, output_dir):
    rq3_rows = []

    for condition in CONDITIONS:
        sd_res = load_eegnet_results(condition, "sd", experiment)
        si_res = load_eegnet_results(condition, "si", experiment)
        tl_res = load_eegnet_results(condition, "tl", experiment)

        if sd_res is None or si_res is None or tl_res is None:
            continue

        sd_acc = get_accuracy_dict(sd_res)
        si_acc = get_accuracy_dict(si_res)
        tl_acc = get_accuracy_dict(tl_res)

        vec_tl, vec_sd, subjects = paired_vectors(tl_acc, sd_acc)
        if len(subjects) >= 2:
            t, p = ttest_rel(vec_tl, vec_sd, alternative="greater")
            rq3_rows.append({"Comparison": "TL vs SD", "Condition": condition, "Mean A (%)": np.mean(vec_tl), "Mean B (%)": np.mean(vec_sd), "t": t, "df": len(subjects) - 1, "p one-tailed": p, "N": len(subjects)})

        vec_tl, vec_si, subjects = paired_vectors(tl_acc, si_acc)
        if len(subjects) >= 2:
            t, p = ttest_rel(vec_tl, vec_si, alternative="greater")
            rq3_rows.append({"Comparison": "TL vs SI", "Condition": condition, "Mean A (%)": np.mean(vec_tl), "Mean B (%)": np.mean(vec_si), "t": t, "df": len(subjects) - 1, "p one-tailed": p, "N": len(subjects)})

        vec_si, vec_sd, subjects = paired_vectors(si_acc, sd_acc)
        if len(subjects) >= 2:
            t, p = ttest_rel(vec_si, vec_sd, alternative="two-sided")
            rq3_rows.append({"Comparison": "SI vs SD", "Condition": condition, "Mean A (%)": np.mean(vec_si), "Mean B (%)": np.mean(vec_sd), "t": t, "df": len(subjects) - 1, "p two-tailed": p, "N": len(subjects)})

    if len(rq3_rows) == 0:
        return

    save_latex_table(
        pd.DataFrame(rq3_rows),
        output_dir,
        "RQ1_3_TL_strategy_comparisons.tex",
        f"{experiment}: paired comparisons between EEGNet training strategies.",
        f"tab:{experiment.lower()}_rq1_3_tl_strategy_comparisons"
    )


# RQ2 SIGNAL SMOOTHING TESTS
# NoPseudo vs PseudoTrainOnly and PseudoTrainAndTest
def make_smoothing_tests(output_dir):
    rows = []

    comparisons = [
        ("AllData_PseudoTrainOnly", "PseudoTrainOnly vs NoPseudo"),
        ("AllData_PseudoTrainAndTest", "PseudoTrainAndTest vs NoPseudo")
    ]

    reference_experiment = "AllData_NoPseudo"

    for strategy in STRATEGIES:
        for condition in CONDITIONS:
            ref_res = load_eegnet_results(condition, strategy, reference_experiment)

            if ref_res is None:
                continue

            ref_acc = get_accuracy_dict(ref_res)

            for test_experiment, comparison_name in comparisons:
                test_res = load_eegnet_results(condition, strategy, test_experiment)

                if test_res is None:
                    continue

                test_acc = get_accuracy_dict(test_res)
                vec_test, vec_ref, subjects = paired_vectors(test_acc, ref_acc)

                if len(subjects) < 2:
                    continue

                t, p = ttest_rel(vec_test, vec_ref, alternative="two-sided")

                rows.append({
                    "Strategy": METHOD_LABELS[strategy],
                    "Condition": condition,
                    "Comparison": comparison_name,
                    "Mean smoothed (%)": np.mean(vec_test),
                    "Mean original (%)": np.mean(vec_ref),
                    "t": t,
                    "df": len(subjects) - 1,
                    "p two-tailed": p,
                    "N": len(subjects)
                })

    if len(rows) == 0:
        print("No smoothing test data found.")
        return

    save_latex_table(
        pd.DataFrame(rows),
        output_dir,
        "RQ2_Smoothing_tests.tex",
        "Paired t-tests comparing smoothed-signal configurations against the original no-pseudo-trial configuration.",
        "tab:rq2_smoothing_tests"
    )


# RQ3 TRAINING DATA SIZE TESTS
# Full data vs reduced source-subject training data
def make_downsampling_tests(output_dir):
    rows = []

    reference_experiment = "AllData_NoPseudo"

    comparisons = [
        ("Downsample_1000_NoPseudo", "FullData vs 1000 trials"),
        ("Downsample_MinSubject_NoPseudo", "FullData vs matched min size")
    ]

    for strategy in ["si", "tl"]:
        for condition in CONDITIONS:
            ref_res = load_eegnet_results(condition, strategy, reference_experiment)

            if ref_res is None:
                continue

            ref_acc = get_accuracy_dict(ref_res)

            for test_experiment, comparison_name in comparisons:
                test_res = load_eegnet_results(condition, strategy, test_experiment)

                if test_res is None:
                    continue

                test_acc = get_accuracy_dict(test_res)
                vec_full, vec_reduced, subjects = paired_vectors(ref_acc, test_acc)

                if len(subjects) < 2:
                    continue

                t, p = ttest_rel(vec_full, vec_reduced, alternative="greater")

                rows.append({
                    "Strategy": METHOD_LABELS[strategy],
                    "Condition": condition,
                    "Comparison": comparison_name,
                    "Mean full data (%)": np.mean(vec_full),
                    "Mean reduced data (%)": np.mean(vec_reduced),
                    "t": t,
                    "df": len(subjects) - 1,
                    "p one-tailed": p,
                    "N": len(subjects)
                })

    if len(rows) == 0:
        print("No downsampling test data found.")
        return

    save_latex_table(
        pd.DataFrame(rows),
        output_dir,
        "RQ3_Downsampling_tests.tex",
        "Paired t-tests comparing full-data SI and TL models against reduced-training-data configurations.",
        "tab:rq3_downsampling_tests"
    )

# RQ4 FINE-TUNING DATA SIZE TESTS
# FT100 vs FT10, FT25, FT50, FT75
def load_finetune_results(condition, fine_tune_percent):
    absolute_dataset_percent = int(round((fine_tune_percent / 100.0) * 70.0))

    path = os.path.join(
        EEGNET_RESULTS_DIR,
        "AllData_NoPseudo",
        f"FT_{absolute_dataset_percent}",
        f"{condition}_tl.npy"
    )

    if not os.path.exists(path):
        print("Missing:", path)
        return None

    return np.load(path, allow_pickle=True).item()


def make_finetuning_tests(output_dir):
    rows = []

    reference_percent = 100
    test_percents = [10, 25, 50, 75]

    for condition in CONDITIONS:
        ref_res = load_finetune_results(condition, reference_percent)

        if ref_res is None:
            continue

        ref_acc = get_accuracy_dict(ref_res)

        for p_ft in test_percents:
            test_res = load_finetune_results(condition, p_ft)

            if test_res is None:
                continue

            test_acc = get_accuracy_dict(test_res)
            vec_100, vec_test, subjects = paired_vectors(ref_acc, test_acc)

            if len(subjects) < 2:
                continue

            t, p = ttest_rel(vec_100, vec_test, alternative="greater")

            rows.append({
                "Condition": condition,
                "Comparison": f"FT100 vs FT{p_ft}",
                "Mean FT100 (%)": np.mean(vec_100),
                f"Mean FT{p_ft} (%)": np.mean(vec_test),
                "t": t,
                "df": len(subjects) - 1,
                "p one-tailed": p,
                "N": len(subjects)
            })

    if len(rows) == 0:
        print("No fine-tuning test data found.")
        return

    save_latex_table(
        pd.DataFrame(rows),
        output_dir,
        "RQ4_Finetuning_tests.tex",
        "Paired t-tests comparing full fine-tuning against reduced fine-tuning data.",
        "tab:rq4_finetuning_tests"
    )

def load_group_mean(condition, strategy, experiment="AllData_NoPseudo"):
    res = load_eegnet_results(condition, strategy, experiment)

    if res is None:
        return None

    if "group_stats" in res:
        return res["group_stats"]["mean_accuracy"] * 100

    accs = list(get_accuracy_dict(res).values())

    if len(accs) == 0:
        return None

    return np.mean(accs)


def make_finetuning_curve(output_dir):
    ft_percents = [10, 25, 50, 75, 100]

    condition_labels = {
        "BSL": "Baseline",
        "SENSORY": "Sensory",
        "DELAY": "Delay"
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, condition in zip(axes, CONDITIONS):
        means = []
        cis = []
        x_values = []

        for p in ft_percents:
            res = load_finetune_results(condition, p)

            if res is None:
                continue

            stats = res["group_stats"]

            means.append(stats["mean_accuracy"] * 100)
            cis.append(stats["ci95"] * 100)
            x_values.append(p)

        means = np.array(means)
        cis = np.array(cis)
        x_values = np.array(x_values)

        ax.plot(
            x_values,
            means,
            marker="o",
            linewidth=2,
            label="TL fine-tuning"
        )

        for x, y in zip(x_values, means):
            ax.text(
                x,
                y + 0.8,
                f"{y:.1f}",
                ha="center",
                fontsize=8
            )

        ax.fill_between(
            x_values,
            means - cis,
            means + cis,
            alpha=0.20
        )

        sd_mean = load_group_mean(condition, "sd")
        si_mean = load_group_mean(condition, "si")

        if sd_mean is not None:
            ax.scatter(
                100,
                sd_mean,
                marker="^",
                s=110,
                color="#8E44AD",
                edgecolor="#8E44AD",
                zorder=5,
                label="SD"
            )

            ax.text(
                103.0,
                sd_mean + 0.2,
                f"{sd_mean:.1f}%",
                va="center",
                fontsize=9,
                color="#8E44AD"
            )

        if si_mean is not None:
            ax.scatter(
                100,
                si_mean,
                marker="s",
                s=90,
                color="#2ECC71",
                edgecolor="#2ECC71",
                zorder=5,
                label="SI"
            )

            ax.text(
                103.0,
                si_mean - 0.2,
                f"{si_mean:.1f}%",
                va="center",
                fontsize=9,
                color="#2ECC71"
            )

        ax.axhline(
            CHANCE,
            linestyle="-.",
            color="black",
            linewidth=1.3
        )

        ax.set_title(condition_labels[condition], fontweight="bold")
        ax.set_xlabel("Fine-tuning data (%)")
        ax.set_xlim(5, 112)
        ax.set_ylim(30, 80)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Classification accuracy (%)")

    legend_elements = [
        Line2D([0], [0], marker="o", linewidth=2, label="TL fine-tuning"),
        Line2D([0], [0], marker="^", color="#8E44AD", linestyle="None", markersize=9, label="SD"),
        Line2D([0], [0], marker="s", color="#2ECC71", linestyle="None", markersize=8, label="SI"),
        Line2D([0], [0], color="black", linestyle="-.", linewidth=1.3, label="Chance level")
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05)
    )

    fig.suptitle(
        "Transfer learning fine-tuning curves with SD and SI references",
        fontweight="bold",
        y=1.12
    )

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        "TL_FinetuneCurve_AllConditions_with_SD_SI.pdf"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)

def make_downsampling_violin(strategy, output_dir):
    print(f"Generating downsampling violin plot for {strategy.upper()}...")

    rows = []

    for experiment, exp_label in DOWNSAMPLE_EXPERIMENTS.items():
        for condition in CONDITIONS:
            res = load_eegnet_results(condition, strategy, experiment)
            if res is None:
                continue

            for sub, acc in get_accuracy_dict(res).items():
                rows.append({
                    "Condition": condition,
                    "Dataset size": exp_label,
                    "Subject": sub,
                    "Accuracy": acc
                })

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"No data found for {strategy.upper()} downsampling plot.")
        return

    experiment_palette = {
        "Matched min size": "#4A90E2",
        "1000 trials": "#E67E22",
        "Full data": "#D5A6E6"
    }

    size_order = ["Matched min size", "1000 trials", "Full data"]

    group_order, group_labels, group_colors = [], [], []

    df["Group"] = df["Condition"].astype(str) + "_" + df["Dataset size"].astype(str)

    for cond in CONDITIONS:
        for size in size_order:
            group = f"{cond}_{size}"
            if group in df["Group"].values:
                group_order.append(group)
                group_labels.append(size)
                group_colors.append(experiment_palette[size])

    group_palette = dict(zip(group_order, group_colors))

    condition_positions = {
        "separators": [2.5, 5.5],
        "labels": [("Baseline", 1), ("Sensory", 4), ("Delay", 7)]
    }

    plot_old_style_violin(
        df=df,
        group_order=group_order,
        group_labels=group_labels,
        group_palette=group_palette,
        title=f"EEGNet {strategy.upper()}: effect of training dataset size",
        save_path=os.path.join(output_dir, f"Downsampling_{strategy.upper()}_Accuracy.pdf"),
        condition_positions=condition_positions
    )


def run_experiment_analysis(experiment, lda_results):
    output_dir = os.path.join(EEGNET_RESULTS_DIR, experiment, "main_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRunning analysis for: {experiment}")

    make_main_violin_plot(experiment, output_dir, lda_results)
    make_individual_accuracy_plots(experiment, output_dir)
    make_confusion_matrices(experiment, output_dir)
    make_rq1_1_table(experiment, output_dir)
    make_rq1_2_table(experiment, output_dir, lda_results)
    make_rq1_3_table(experiment, output_dir)


print("Loading LDA results...")
with open(LDA_RESULTS_PATH, "rb") as f:
    lda_results = pickle.load(f)

for experiment in EXPERIMENTS:
    run_experiment_analysis(experiment, lda_results)

downsample_output = os.path.join(EEGNET_RESULTS_DIR, "downsampling_analysis")
os.makedirs(downsample_output, exist_ok=True)

make_downsampling_violin("tl", downsample_output)
make_downsampling_violin("si", downsample_output)

rq4_output = os.path.join(EEGNET_RESULTS_DIR, "rq4_analysis")
os.makedirs(rq4_output, exist_ok=True)
make_smoothing_tests(rq4_output)
make_downsampling_tests(rq4_output)
make_finetuning_tests(rq4_output)
make_finetuning_curve(rq4_output)


print("\nAnalysis complete.")
print("Output folders are inside:")
print(EEGNET_RESULTS_DIR)