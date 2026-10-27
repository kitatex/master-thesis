import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from config.paths import TOPICS_PATH

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
RESULTS_DIR = (TOPICS_PATH / "_polarization_results").resolve()
INPUT_CSV = RESULTS_DIR / "polarization_simulations_n1000.csv"

# Two separate PNG outputs for easy LaTeX integration
OUTPUT_PLOT_TOPICS = RESULTS_DIR / "fig2_topic_level_breakdown.png"
OUTPUT_PLOT_DELTA = RESULTS_DIR / "fig3_discriminative_gap_delta.png"

METRICS = {
    "ac": "AC",
    "aei": "AEI",
    "rwc": "RWC",
    "r_pb_seed_only": "OPC",
}

GROUP_POL = "Political"
GROUP_CTRL = "Non-Controversial (Control)"

# Set to False if you want just the clean bars without Standard Error whiskers
SHOW_SE_BARS_ON_DELTA = True


def compute_simple_differences(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for m_key, m_label in METRICS.items():
        if m_key not in df.columns:
            continue

        # 1. Average across the 100 simulation runs per topic first
        topic_df = df.groupby(["group", "topic"], as_index=False)[m_key].mean()
        pol_vals = topic_df[topic_df["group"] == GROUP_POL][m_key].values
        ctrl_vals = topic_df[topic_df["group"] == GROUP_CTRL][m_key].values

        pol_mean = float(np.mean(pol_vals))
        ctrl_mean = float(np.mean(ctrl_vals))
        raw_diff = pol_mean - ctrl_mean

        # Standard Error of the difference between two means (Welch's formula)
        pol_var = np.var(pol_vals, ddof=1)
        ctrl_var = np.var(ctrl_vals, ddof=1)
        se_diff = float(np.sqrt(pol_var / len(pol_vals) + ctrl_var / len(ctrl_vals)))

        records.append(
            {
                "metric_key": m_key,
                "Metric": m_label,
                "pol_mean": pol_mean,
                "ctrl_mean": ctrl_mean,
                "raw_diff": raw_diff,
                "se_diff": se_diff,
            }
        )

    return pd.DataFrame(records)


def plot_topic_breakdown(df: pd.DataFrame):
    """
    PLOT 1: Per-Topic Breakdown (4 columns, 1 for each metric).
    Shows the exact Mean ± SD (across the 100 runs) for all 12 topics,
    plus the vertical Group Mean lines.
    """
    df = df.copy()
    df["topic"] = df["topic"].str.replace(r"^0 ", "", regex=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    available_keys = [k for k in METRICS.keys() if k in df.columns]

    fig, axes = plt.subplots(1, len(available_keys), figsize=(13, 5.5), sharey=True)
    palette = {GROUP_POL: "#171717", GROUP_CTRL: "#9a9999"}

    # Order topics: Political on top (sorted by mean r_pb), Control on bottom
    rpb_means = df.groupby(["group", "topic"])["r_pb_seed_only"].mean().reset_index()
    pol_order = (
        rpb_means[rpb_means["group"] == GROUP_POL]
        .sort_values("r_pb_seed_only", ascending=False)["topic"]
        .tolist()
    )
    ctrl_order = (
        rpb_means[rpb_means["group"] == GROUP_CTRL]
        .sort_values("r_pb_seed_only", ascending=False)["topic"]
        .tolist()
    )
    topic_order = pol_order + ctrl_order

    for col_idx, m_key in enumerate(available_keys):
        ax = axes[col_idx]
        m_label = METRICS[m_key]

        # Plot topic mean ± SD across the 100 stochastic runs
        sns.pointplot(
            data=df,
            x=m_key,
            y="topic",
            hue="group",
            order=topic_order,
            palette=palette,
            errorbar="sd",
            linestyle="none",
            markers="o",
            markersize=6,
            err_kws={"linewidth": 1.5},
            legend=False,
            ax=ax,
        )

        # Group means
        pol_mean = df[df["group"] == GROUP_POL][m_key].mean()
        ctrl_mean = df[df["group"] == GROUP_CTRL][m_key].mean()
        diff = pol_mean - ctrl_mean

        ax.axvline(pol_mean, color=palette[GROUP_POL], linestyle="--", linewidth=1.6)
        ax.axvline(ctrl_mean, color=palette[GROUP_CTRL], linestyle=":", linewidth=1.6)

        # Subtle shading between the Political Mean and Control Mean
        gap_color = "#2ca25f" if diff > 0 else "#de2d26"
        ax.axvspan(
            min(pol_mean, ctrl_mean),
            max(pol_mean, ctrl_mean),
            color=gap_color,
            alpha=0.12,
        )

        # Horizontal line separating Political (top 6) and Control (bottom 6)
        ax.axhline(
            len(pol_order) - 0.5, color="black", linestyle="-", linewidth=1.0, alpha=0.5
        )

        ax.set_title(rf"{m_label} ($\Delta = {diff:+.2f}$)", fontweight="bold")
        ax.set_xlabel(r"Score (Mean $\pm$ SD)")
        ax.set_xlim(-0.05, 1.05)

        ax.set_ylabel("")

    # Add a single clean legend at the top of the figure
    pol_dot = mlines.Line2D(
        [],
        [],
        color=palette[GROUP_POL],
        marker="o",
        linestyle="None",
        markersize=7,
        label=r"Political Topic (Mean $\pm$ SD)",
    )
    ctrl_dot = mlines.Line2D(
        [],
        [],
        color=palette[GROUP_CTRL],
        marker="o",
        linestyle="None",
        markersize=7,
        label=r"Non-Political Topic (Mean $\pm$ SD)",
    )
    pol_line = mlines.Line2D(
        [],
        [],
        color=palette[GROUP_POL],
        linestyle="--",
        linewidth=1.6,
        label="Political Topics Mean",
    )
    ctrl_line = mlines.Line2D(
        [],
        [],
        color=palette[GROUP_CTRL],
        linestyle=":",
        linewidth=1.6,
        label="Non-Political Topics Mean",
    )

    fig.legend(
        handles=[pol_dot, ctrl_dot, pol_line, ctrl_line],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        frameon=True,
        edgecolor="black",
    )

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PLOT_TOPICS, "wb") as f:
        fig.savefig(f, format="png", dpi=300, bbox_inches="tight")

    print(f"Saved Plot 1 (Per-Topic Breakdown) to:\n{OUTPUT_PLOT_TOPICS}")
    plt.show()


def plot_mean_difference_bar(diff_df: pd.DataFrame):
    """
    PLOT 2: Simple, pragmatic bar chart of the Mean Difference
    (Delta = Political Mean - Control Mean) for each metric.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    y_pos = np.arange(len(diff_df))
    # Highlight positive discriminative gap in orange/green, near-zero/negative in grey
    bar_colors = [
        "#d95f02" if row["metric_key"] == "r_pb_seed_only" else "#888888"
        for _, row in diff_df.iterrows()
    ]

    xerr = diff_df["se_diff"] if SHOW_SE_BARS_ON_DELTA else None

    ax.barh(
        y_pos,
        diff_df["raw_diff"],
        xerr=xerr,
        color=bar_colors,
        edgecolor="black",
        height=0.52,
        alpha=0.9,
        capsize=4 if SHOW_SE_BARS_ON_DELTA else 0,
    )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(diff_df["Metric"], fontweight="bold")
    ax.invert_yaxis()

    xlabel = (
        r"Mean Difference: $\Delta = \mu_{\mathrm{political}} - \mu_{\mathrm{nonpolitical}}$ ($\pm$ SE)"
        if SHOW_SE_BARS_ON_DELTA
        else r"Mean Difference: $\Delta = \mu_{\mathrm{political}} - \mu_{\mathrm{nonpolitical}}$"
    )

    ax.set_xlabel(xlabel, fontweight="bold")

    # Add clean text labels next to each bar
    min_x = min(0.0, (diff_df["raw_diff"] - diff_df["se_diff"]).min())
    max_x = max(0.35, (diff_df["raw_diff"] + diff_df["se_diff"]).max())
    ax.set_xlim(min_x - 0.12, max_x + 0.12)

    for i, row in diff_df.iterrows():
        val = row["raw_diff"]
        se = row["se_diff"] if SHOW_SE_BARS_ON_DELTA else 0.0
        offset = (se + 0.015) if val >= 0 else -(se + 0.015)
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + offset,
            i,  # type: ignore
            f"{val:+.3f}",
            va="center",
            ha=ha,
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PLOT_DELTA, "wb") as f:
        fig.savefig(f, format="png", dpi=300, bbox_inches="tight")

    print(f"Saved Plot 2 (Mean Difference Bar Chart) to:\n{OUTPUT_PLOT_DELTA}")
    plt.show()


def main():
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    diff_df = compute_simple_differences(df)

    # Generate the two standalone PNGs
    plot_topic_breakdown(df)
    plot_mean_difference_bar(diff_df)


if __name__ == "__main__":
    main()
