import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config.paths import TOPICS_PATH

TOPIC_NAME = "#aiethics"

INPUT_DIR = TOPICS_PATH / f"{TOPIC_NAME}/eda/url_eda.csv"

OUTPUT_PATH = TOPICS_PATH / f"{TOPIC_NAME}/eda/mbfc_eda_distribution.png"


def columns_count(matched_csv_path):
    df = pd.read_csv(matched_csv_path)
    matched_df = df.dropna(subset=["bias_score", "factual_reporting"]).copy()
    matched_df["bias_score_str"] = matched_df["bias_score"].astype(str)

    sns.set_theme(style="whitegrid")
    # Increased figsize slightly to accommodate larger gaps
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    axes = axes.flatten()

    # 1. Ideological Bias
    bias_colors = {
        "-1.0": "#2b5c8f",
        "-0.5": "#74a9cf",
        "0.0": "#cccccc",
        "0.5": "#fc9272",
        "1.0": "#de2d26",
    }
    sns.countplot(
        data=matched_df,
        x="bias_score_str",
        ax=axes[0],
        palette=bias_colors,
        order=["-1.0", "-0.5", "0.0", "0.5", "1.0"],
    )
    axes[0].set_title(
        "Ideological Bias Distribution", fontweight="bold", fontsize=15, pad=15
    )
    axes[0].set_xticklabels(["Left", "L-Center", "Neutral", "R-Center", "Right"])
    axes[0].set_xlabel("Bias Scale", fontsize=12)

    # 2. Factual Reporting
    sns.countplot(
        data=matched_df,
        x="factual_reporting",
        ax=axes[1],
        palette="viridis",
        order=["high", "mixed", "low"],
    )
    axes[1].set_title(
        "Factual Reporting Quality", fontweight="bold", fontsize=15, pad=15
    )
    axes[1].set_xlabel("Factual Label", fontsize=12)

    # 3. Credibility Rating
    sns.countplot(data=matched_df, x="credibility", ax=axes[2], palette="magma")
    axes[2].set_title("MBFC Credibility Rating", fontweight="bold", fontsize=15, pad=15)
    # Rotation prevents label overlap here
    axes[2].tick_params(axis="x", rotation=30)

    # 4. Media Type (Top 5)
    top_media = matched_df["media_type"].value_counts().nlargest(5).index
    sns.countplot(
        data=matched_df[matched_df["media_type"].isin(top_media)],
        x="media_type",
        ax=axes[3],
        palette="Set2",
    )
    axes[3].set_title("Top 5 Media Types", fontweight="bold", fontsize=15, pad=15)
    axes[3].tick_params(axis="x", rotation=20)

    # 5. Popularity/Traffic
    pop_order = ["high traffic", "medium traffic", "low traffic"]
    sns.countplot(
        data=matched_df,
        x="popularity",
        ax=axes[4],
        palette="coolwarm",
        order=[p for p in pop_order if p in matched_df["popularity"].unique()],
    )
    axes[4].set_title(
        "Source Popularity (Traffic)", fontweight="bold", fontsize=15, pad=15
    )

    # 6. Country (Top 5)
    top_countries = matched_df["country"].value_counts().nlargest(5).index
    sns.countplot(
        data=matched_df[matched_df["country"].isin(top_countries)],
        x="country",
        ax=axes[5],
        palette="tab10",
    )
    axes[5].set_title(
        "Top 5 Countries of Origin", fontweight="bold", fontsize=15, pad=15
    )

    # --- THE FIX: Precise control over spacing ---
    # hspace is vertical spacing, wspace is horizontal
    plt.subplots_adjust(
        left=0.07, bottom=0.1, right=0.95, top=0.92, wspace=0.35, hspace=0.45
    )

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Enriched visualization saved to {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    columns_count(INPUT_DIR)
