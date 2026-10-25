import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from config.paths import TOPICS_PATH

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
RESULTS_DIR = (TOPICS_PATH / "_polarization_results").resolve()
INPUT_CSV = RESULTS_DIR / "polarization_simulations_n1000.csv"
OUTPUT_PLOT_GARIMELLA = RESULTS_DIR / "fig1_garimella_split_violins.png"
OUTPUT_STATS_CSV = RESULTS_DIR / "table_metric_comparison_summary.csv"

# Map CSV columns to clean Thesis labels
# (Uncomment "r_pb_seed_only" if you also want a 5th violin for Seed-Only OPC!)
METRICS = {
    "ac": "AC",
    "aei": "AEI",
    "rwc": "RWC",
    "r_pb_seed_only": "OPC",
}

GROUP_POL = "Political"
GROUP_CTRL = "Non-Controversial (Control)"


def prepare_long_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Melts the simulation DataFrame into long format for across-metric plotting,
    both at the simulation level (n=1200) and aggregated topic-mean level (n=12).
    """
    available_metrics = [m for m in METRICS.keys() if m in df.columns]

    df_long = df.melt(
        id_vars=["group", "topic", "simulation"],
        value_vars=available_metrics,
        var_name="metric_key",
        value_name="score",
    )
    df_long["Metric"] = df_long["metric_key"].map(METRICS)

    # Compute topic-level means (12 topics x M metrics) to avoid pseudoreplication
    df_topic_means = df_long.groupby(
        ["group", "topic", "metric_key", "Metric"], as_index=False
    )["score"].agg(topic_mean="mean", topic_sd="std")
    return df_long, df_topic_means  # type: ignore


def export_summary_statistics(df_long: pd.DataFrame, df_topic_means: pd.DataFrame):
    """
    Computes and prints a comprehensive statistical table comparing Political vs. Control
    for each polarization measure (Mean, SD, Difference, Cohen's d, Welch's p, Mann-Whitney p).
    """
    rows = []
    print("\n" + "=" * 100)
    print(
        f"{'Metric':<18} | {'Political Mean (SD)':<21} | {'Control Mean (SD)':<21} | "
        f"{'Diff (Δ)':<10} | {'Cohen d':<9} | {'p-val (Topic)'}"
    )
    print("-" * 100)

    for m_key, m_label in METRICS.items():
        if m_key not in df_topic_means["metric_key"].unique():
            continue

        clean_name = m_label.replace("\n", " ")
        pol_topics = df_topic_means[
            (df_topic_means["metric_key"] == m_key)
            & (df_topic_means["group"] == GROUP_POL)
        ]["topic_mean"].values
        ctrl_topics = df_topic_means[
            (df_topic_means["metric_key"] == m_key)
            & (df_topic_means["group"] == GROUP_CTRL)
        ]["topic_mean"].values

        pol_m, pol_s = np.mean(pol_topics), np.std(pol_topics, ddof=1)  # type: ignore
        ctrl_m, ctrl_s = np.mean(ctrl_topics), np.std(ctrl_topics, ddof=1)  # type: ignore
        diff = pol_m - ctrl_m

        # Pooled SD for Cohen's d across the 12 independent topics (n1=6, n2=6)
        pooled_sd = np.sqrt(
            ((len(pol_topics) - 1) * pol_s**2 + (len(ctrl_topics) - 1) * ctrl_s**2)
            / (len(pol_topics) + len(ctrl_topics) - 2)
        )
        cohen_d = diff / pooled_sd if pooled_sd > 1e-12 else 0.0

        # Topic-level Welch's t-test and Mann-Whitney U test (one-sided: Political > Control)
        _, p_welch = stats.ttest_ind(
            pol_topics, ctrl_topics, equal_var=False, alternative="greater"
        )
        _, p_mwu = stats.mannwhitneyu(pol_topics, ctrl_topics, alternative="greater")

        # Within-topic stochastic stability (mean SD across the 100 seeds)
        within_seed_sd = df_topic_means[df_topic_means["metric_key"] == m_key][
            "topic_sd"
        ].mean()

        print(
            f"{clean_name:<18} | {pol_m:6.3f} (±{pol_s:5.3f})      | {ctrl_m:6.3f} (±{ctrl_s:5.3f})      | "
            f"{diff:+8.3f}  | {cohen_d:+7.2f}  | {p_welch:.4f} (MWU: {p_mwu:.4f})"
        )

        rows.append(
            {
                "metric": clean_name,
                "political_mean": round(pol_m, 4),
                "political_between_topic_sd": round(pol_s, 4),
                "control_mean": round(ctrl_m, 4),
                "control_between_topic_sd": round(ctrl_s, 4),
                "mean_within_topic_seed_sd": round(within_seed_sd, 4),
                "difference_pol_minus_ctrl": round(diff, 4),
                "cohens_d_topic_level": round(cohen_d, 4),
                "p_value_welch_one_sided": round(float(p_welch), 5),  # type: ignore
                "p_value_mwu_one_sided": round(float(p_mwu), 5),
            }
        )

    print("=" * 100 + "\n")
    summary_df = pd.DataFrame(rows)
    with open(OUTPUT_STATS_CSV, "w", encoding="utf-8", newline="") as f:
        summary_df.to_csv(f, index=False)
    print(f"Saved statistical summary table to:\n{OUTPUT_STATS_CSV}")


def plot_garimella_beanplot(df_long: pd.DataFrame, df_topic_means: pd.DataFrame):
    """
    Replicates the Garimella et al. (2018) split-violin ('beanplot') visualization:
      - Left half-violin (Black/Charcoal): Political (Controversial)
      - Right half-violin (Light Grey): Non-Controversial (Control)
      - Protruding horizontal lines: Group means
      - Center spine tick marks: Individual topic means (n=6 per side)
    """
    sns.set_theme(style="ticks", context="paper", font_scale=1.25)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    metric_order = [
        METRICS[k] for k in METRICS.keys() if k in df_long["metric_key"].unique()
    ]
    palette = {GROUP_POL: "#222222", GROUP_CTRL: "#cccccc"}

    # 1. Draw Split Violins
    sns.violinplot(
        data=df_long,
        x="Metric",
        y="score",
        hue="group",
        order=metric_order,
        hue_order=[GROUP_POL, GROUP_CTRL],
        split=True,
        inner=None,
        palette=palette,
        cut=1,
        bw_adjust=0.8,
        linewidth=0.8,
        ax=ax,
    )

    # 2. Overlay Garimella-style Protruding Mean Lines & Topic-Mean Spine Ticks
    for i, m_label in enumerate(metric_order):
        sub_topics = df_topic_means[df_topic_means["Metric"] == m_label]

        pol_vals = sub_topics[sub_topics["group"] == GROUP_POL]["topic_mean"].values
        ctrl_vals = sub_topics[sub_topics["group"] == GROUP_CTRL]["topic_mean"].values

        # Protruding horizontal group mean lines (Left for Political, Right for Control)
        if len(pol_vals) > 0:
            pol_mean = np.mean(pol_vals)  # type: ignore
            ax.plot(
                [i - 0.38, i],
                [pol_mean, pol_mean],
                color="black",
                linewidth=2.2,
                zorder=5,
            )
            # White tick marks along the left spine for each of the 6 Political topic means
            for v in pol_vals:
                ax.plot(
                    [i - 0.04, i - 0.005],
                    [v, v],
                    color="white",
                    linewidth=1.3,
                    zorder=6,
                )

        if len(ctrl_vals) > 0:
            ctrl_mean = np.mean(ctrl_vals)  # type: ignore
            ax.plot(
                [i, i + 0.38],
                [ctrl_mean, ctrl_mean],
                color="black",
                linewidth=2.2,
                zorder=5,
            )
            # Dark tick marks along the right spine for each of the 6 Control topic means
            for v in ctrl_vals:
                ax.plot(
                    [i + 0.005, i + 0.04],
                    [v, v],
                    color="#222222",
                    linewidth=1.3,
                    zorder=6,
                )

    # Reference zero line
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.2, alpha=0.7)

    ax.set_ylabel("Controversy Score", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylim(-0.15, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Clean custom legend matching Garimella et al.
    pol_patch = mpatches.Patch(
        facecolor="#222222", edgecolor="black", label="Political"
    )
    ctrl_patch = mpatches.Patch(
        facecolor="#cccccc", edgecolor="black", label="Non-Political"
    )
    mean_line = plt.Line2D([0], [0], color="black", linewidth=2.2, label="Group Mean")  # type: ignore
    ax.legend(
        handles=[pol_patch, ctrl_patch, mean_line],
        loc="upper right",
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PLOT_GARIMELLA, "wb") as f:
        fig.savefig(f, format="png", dpi=300, bbox_inches="tight")

    print(f"Saved Garimella-style split beanplot to:\n{OUTPUT_PLOT_GARIMELLA}")
    plt.show()


def main():
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found. Wait for simulations to finish first.")
        return

    df = pd.read_csv(INPUT_CSV)
    df_long, df_topic_means = prepare_long_data(df)
    export_summary_statistics(df_long, df_topic_means)
    plot_garimella_beanplot(df_long, df_topic_means)


if __name__ == "__main__":
    main()
