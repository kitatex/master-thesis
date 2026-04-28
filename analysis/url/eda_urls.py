import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.paths import TOPICS_PATH

TOPIC_NAME = "#gamedev"

INPUT_DIR = TOPICS_PATH / f"{TOPIC_NAME}/eda/url_eda.csv"


def visualize_eda(matched_csv_path):
    # Load the processed data
    df = pd.read_csv(matched_csv_path)

    # Filter only rows where we successfully matched an MBFC score
    matched_df = df.dropna(subset=["bias_score", "factual_reporting"]).copy()

    # --- FIX: Cast scores to strings to match dictionary keys ---
    matched_df["bias_score_str"] = matched_df["bias_score"].astype(str)

    # Set plot style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Ideological Bias Distribution ---
    # Update keys to strings to prevent the ValueError
    bias_colors = {
        "-1.0": "#2b5c8f",
        "-0.5": "#74a9cf",
        "0.0": "#cccccc",
        "0.5": "#fc9272",
        "1.0": "#de2d26",
    }

    # Use the new string-based order
    bias_order = ["-1.0", "-0.5", "0.0", "0.5", "1.0"]

    sns.countplot(
        data=matched_df,
        x="bias_score_str",  # Use the string column here
        ax=axes[0],
        palette=bias_colors,
        order=bias_order,
    )

    axes[0].set_title("Distribution of Ideological Bias (-1 to 1)", fontsize=14, pad=10)
    axes[0].set_xlabel("Bias Score (Left to Right)", fontsize=12)
    axes[0].set_ylabel("Number of Shared URLs", fontsize=12)
    axes[0].set_xticklabels(
        [
            "-1.0\n(Left)",
            "-0.5\n(Left-Center)",
            "0.0\n(Neutral)",
            "0.5\n(Right-Center)",
            "1.0\n(Right)",
        ]
    )

    # --- Plot 2: Factual Reporting Distribution ---
    factual_order = ["high", "mixed", "low"]
    factual_colors = {"high": "#31a354", "mixed": "#fd8d3c", "low": "#e6550d"}

    sns.countplot(
        data=matched_df,
        x="factual_reporting",
        ax=axes[1],
        palette=factual_colors,
        order=factual_order,
    )

    axes[1].set_title("Distribution of Factual Reporting Quality", fontsize=14, pad=10)
    axes[1].set_xlabel("Factual Reporting Label", fontsize=12)
    axes[1].set_ylabel("Number of Shared URLs", fontsize=12)

    plt.tight_layout()
    plt.savefig("mbfc_eda_distribution.png", dpi=300)
    print("Visualization saved successfully as 'mbfc_eda_distribution.png'!")
    plt.show()


if __name__ == "__main__":
    visualize_eda(INPUT_DIR)
