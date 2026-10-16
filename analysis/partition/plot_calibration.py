import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.paths import TOPICS_PATH

# ==========================================
# PATHS & CONFIGURATION
# ==========================================
TUNING_DIR = (TOPICS_PATH / "_hyperparameter_tuning").resolve()
INPUT_CSV = TUNING_DIR / "unsupervised_leiden_calibration.csv"
SUMMARY_CSV = TUNING_DIR / "unsupervised_leiden_summary_table.csv"
OUTPUT_PLOT = TUNING_DIR / "unsupervised_leiden_calibration_plot.png"


def compute_optimal_gamma(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
    Computes the optimal resolution parameter on the Political corpus by maximizing
    the mean topological quality penalized by within-topic stochastic instability.
    """
    pol_df = df[df["group"] == "Political"]

    # First compute mean and SD across seeds WITHIN each (topic, resolution)
    topic_res = (
        pol_df.groupby(["topic", "resolution"])["topo_quality"]
        .agg(topic_mean="mean", within_seed_std="std")
        .reset_index()
    )
    topic_res["within_seed_std"] = topic_res["within_seed_std"].fillna(0.0)

    # Then aggregate across topics for each resolution
    agg_topo = topic_res.groupby("resolution").agg(
        mean_topo=("topic_mean", "mean"),
        between_topic_std=("topic_mean", "std"),
        mean_within_seed_std=("within_seed_std", "mean"),
    )
    # Penalize by within-topic stochastic variance across Leiden seeds
    agg_topo["penalized_score"] = (
        agg_topo["mean_topo"] - 0.5 * agg_topo["mean_within_seed_std"]
    )
    best_gamma = float(agg_topo["penalized_score"].idxmax())  # type: ignore
    return best_gamma, agg_topo


def print_and_export_summary(df: pd.DataFrame, best_gamma: float):
    """
    Prints a thesis-ready summary table and exports it to CSV.
    """
    pol_df = df[df["group"] == "Political"]
    ctrl_df = df[df["group"] == "Non-Controversial (Control)"]
    resolutions = sorted(df["resolution"].unique())

    summary_rows = []
    print("\n" + "=" * 96)
    print(
        f"{'Gamma':<6} | {'Topo Score (Pol)':<18} | {'Seed SD':<8} | "
        f"{'Min Ratio':<10} | {'r_pb (Pol All)':<16} | {'r_pb (Pol Seed)':<15} | {'r_pb (Control)':<14}"
    )
    print("-" * 96)

    for res in resolutions:
        p_sub = pol_df[pol_df["resolution"] == res]
        c_sub = ctrl_df[ctrl_df["resolution"] == res]
        if p_sub.empty:
            continue

        # Within-topic seed SD (measures stochastic stability of Leiden)
        within_seed_sd = p_sub.groupby("topic")["topo_quality"].std().fillna(0.0).mean()

        topo_m = p_sub["topo_quality"].mean()
        topo_s = p_sub["topo_quality"].std()
        min_r = p_sub["minority_ratio"].mean()
        r_all_m = p_sub["r_pb"].mean()
        r_all_s = p_sub["r_pb"].std()
        r_seed_m = (
            p_sub["r_pb_seed_only"].mean()
            if "r_pb_seed_only" in p_sub.columns
            else float("nan")
        )
        r_ctrl_m = c_sub["r_pb"].mean() if not c_sub.empty else float("nan")

        marker = " *" if np.isclose(res, best_gamma) else ""
        print(
            f"{res:<6.2f} | {topo_m:.4f} (±{topo_s:.4f}) | {within_seed_sd:<8.4f} | "
            f"{min_r:<10.3f} | {r_all_m:.4f} (±{r_all_s:.4f}) | "
            f"{r_seed_m:<15.4f} | {r_ctrl_m:<14.4f}{marker}"
        )

        summary_rows.append(
            {
                "resolution": res,
                "pol_topo_mean": round(topo_m, 4),
                "pol_topo_between_sd": round(topo_s, 4),
                "pol_topo_within_seed_sd": round(within_seed_sd, 4),
                "pol_minority_ratio": round(min_r, 4),
                "pol_r_pb_all_mean": round(r_all_m, 4),
                "pol_r_pb_all_sd": round(r_all_s, 4),
                "pol_r_pb_seed_mean": round(r_seed_m, 4),
                "ctrl_r_pb_all_mean": round(r_ctrl_m, 4),
            }
        )

    print("=" * 96)
    print(f"* Optimal Unsupervised Topological Resolution: gamma = {best_gamma}\n")

    summary_df = pd.DataFrame(summary_rows)
    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        summary_df.to_csv(f, index=False)
    print(f"Saved summary table CSV to:\n{SUMMARY_CSV}")


def plot_unsupervised_calibration(df: pd.DataFrame, best_gamma: float):
    resolutions = sorted(df["resolution"].unique())

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    palette = {"Political": "#d95f02", "Non-Controversial (Control)": "#7570b3"}

    # Note: errorbar=("ci", 95) plots the 95% confidence interval of the mean,
    # which is visually much cleaner than raw across-topic SD when topics differ in scale.
    # --- Panel 1: Unsupervised Topological Objective (RWC x Balance) ---
    sns.lineplot(
        data=df,
        x="resolution",
        y="topo_quality",
        hue="group",
        palette=palette,
        linewidth=2.8,
        markers=True,
        errorbar=("ci", 95),
        ax=ax1,
    )
    ax1.axvline(
        best_gamma,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=rf"Topological Optimum ($\gamma = {best_gamma}$)",
    )
    ax1.set_title(
        r"Step 1 (Unsupervised Selection): Topological Partition Quality ($\mathrm{RWC} \times \mathrm{Balance}$)",
        fontweight="bold",
    )
    ax1.set_ylabel("Topological Score (95% CI)")
    ax1.legend(loc="upper right")

    # --- Panel 2: Partition Balance (Whisker-Cut Prevention) ---
    sns.lineplot(
        data=df,
        x="resolution",
        y="minority_ratio",
        hue="group",
        palette=palette,
        linewidth=2.5,
        markers=True,
        errorbar=("ci", 95),
        legend=False,
        ax=ax2,
    )
    ax2.axvline(best_gamma, color="crimson", linestyle="--", linewidth=1.8)
    if not np.isclose(best_gamma, 0.3):
        ax2.axvline(0.3, color="black", linestyle=":", linewidth=1.5)
    ax2.set_title(
        "Step 2 (Unsupervised Diagnostic): Macroscopic Partition Balance (Avoiding Whisker Cuts)",
        fontweight="bold",
    )
    ax2.set_ylabel(r"Minority Ratio ($\min(|S_0|, |S_1|)/N$)")
    ax2.set_ylim(0, 0.55)

    # --- Panel 3: Downstream Ideological Validation (Point-Biserial r_pb) ---
    sns.lineplot(
        data=df,
        x="resolution",
        y="r_pb",
        hue="group",
        palette=palette,
        linewidth=2.8,
        markers=True,
        errorbar=("ci", 95),
        legend=False,
        ax=ax3,
    )
    ax3.axvline(best_gamma, color="crimson", linestyle="--", linewidth=1.8)
    if not np.isclose(best_gamma, 0.3):
        ax3.axvline(0.3, color="black", linestyle=":", linewidth=1.5)
    ax3.set_title(
        r"Step 3 (Out-of-Loop Validation): Ideological Alignment ($r_{pb}$) Across $\gamma$",
        fontweight="bold",
    )
    ax3.set_xlabel(r"Leiden Resolution Parameter ($\gamma$)")
    ax3.set_ylabel("Point-Biserial Correlation ($r_{pb}$)")
    ax3.set_xticks(resolutions)

    plt.tight_layout()

    # Safe Windows binary write to prevent PIL OSError [Errno 22]
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PLOT, "wb") as f:
        fig.savefig(f, format="png", dpi=300, bbox_inches="tight")

    print(f"Saved 3-panel calibration plot to:\n{OUTPUT_PLOT}")
    plt.show()


def main():
    if not INPUT_CSV.exists():
        print(f"Error: Simulation CSV not found at:\n{INPUT_CSV}")
        print("Run tune_leiden_unsupervised.py first to generate the data.")
        return

    print(f"Loading simulation results from:\n{INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    best_gamma, _ = compute_optimal_gamma(df)
    print_and_export_summary(df, best_gamma)
    plot_unsupervised_calibration(df, best_gamma)


if __name__ == "__main__":
    main()
