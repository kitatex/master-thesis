import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from config.paths import TOPICS_PATH

TOPIC_NAME = "#ukraine"
MIN_RP = 2
MIN_URL = 3

INPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/community/{TOPIC_NAME}_minrp{MIN_RP}_minurl{MIN_URL}_itp_cd.graphml"
)
URL_EDA_CSV = TOPICS_PATH / f"{TOPIC_NAME}/eda/url_eda.csv"

OUTPUT_PLOT = TOPICS_PATH / f"{TOPIC_NAME}/eda/{TOPIC_NAME}_community_profiles.png"
OUTPUT_CSV = TOPICS_PATH / f"{TOPIC_NAME}/eda/{TOPIC_NAME}_community_stats.csv"


def main():
    print(f"Loading community graph: {INPUT_GRAPHML.name}")
    G = nx.read_graphml(INPUT_GRAPHML)

    # 1. Extract Node Data
    node_data = []
    for node, data in G.nodes(data=True):
        raw_id = data.get("name", data.get("v_name", node))
        try:
            clean_id = str(int(float(raw_id)))
        except (ValueError, TypeError):
            clean_id = str(raw_id).strip()

        comm = data.get("leiden_community")
        opinion = data.get("interpolated_bias_score")

        if comm is not None and comm != "unknown" and opinion is not None:
            node_data.append(
                {"user_id": clean_id, "community": str(comm), "opinion": float(opinion)}
            )

    df_nodes = pd.DataFrame(node_data)
    print(
        f"Extracted data for {len(df_nodes)} nodes across {df_nodes['community'].nunique()} total communities."
    )

    # Filter to Top 20 Largest Communities
    # Count the sizes of each community, grab the top 20, and convert their IDs to a list
    top_communities = df_nodes["community"].value_counts().nlargest(20).index.tolist()

    # Filter the dataframe to only include nodes inside those top communities
    df_nodes = df_nodes[df_nodes["community"].isin(top_communities)]
    print(
        f"Filtered to Top {len(top_communities)} communities (Remaining nodes: {len(df_nodes)})."
    )

    # 2. Sort communities by Median Opinion (Left to Right)
    comm_medians = df_nodes.groupby("community")["opinion"].median().sort_values()
    comm_order = comm_medians.index.tolist()

    # 3. Load URL Metadata and Merge
    print(f"Loading URL metadata from {URL_EDA_CSV.name}...")
    df_urls = pd.read_csv(URL_EDA_CSV)
    df_urls["user_id"] = df_urls["user_id"].astype(str).str.strip()

    df_merged = df_urls.merge(
        df_nodes[["user_id", "community"]], on="user_id", how="inner"
    )
    print(f"Matched {len(df_merged)} URL shares to the top community members.")

    # 4. Calculate Community-Level Stats
    stats = []
    for comm in comm_order:
        comm_urls = df_merged[df_merged["community"] == comm]
        n_urls = len(comm_urls)
        n_users = len(df_nodes[df_nodes["community"] == comm])

        if n_urls > 0:
            pct_high_fact = (
                (comm_urls["factual_reporting"].str.lower() == "high").sum()
                / n_urls
                * 100
            )

            cred_col = comm_urls["credibility"].fillna("").str.lower()
            pct_mixed_cred = (
                cred_col.str.contains("medium credibility").sum() / n_urls * 100
            )

            top_media = comm_urls["media_type"].mode()
            top_media_val = top_media.iloc[0] if not top_media.empty else "N/A"
        else:
            pct_high_fact = 0.0
            pct_mixed_cred = 0.0
            top_media_val = "N/A"

        stats.append(
            {
                "Community": comm,
                "Users": n_users,
                "Median Opinion": comm_medians[comm],
                "% High Factual": pct_high_fact,
                "% Mixed Credibility": pct_mixed_cred,
                "Top Media Type": top_media_val,
            }
        )

    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved numerical statistics to {OUTPUT_CSV.name}")

    # 5. Advanced Visualization
    print("Generating Multi-Panel Visualization...")
    sns.set_theme(style="whitegrid")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(16, 16), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]}
    )

    # --- Plot 1: Opinion Distribution ---
    sns.boxplot(
        data=df_nodes,
        x="community",
        y="opinion",
        order=comm_order,
        ax=ax1,
        color="whitesmoke",
        linewidth=1.5,
    )
    ax1.set_title(
        f"Top Community Profiles ({TOPIC_NAME}) - Sorted by Ideology",
        fontsize=18,
        fontweight="bold",
    )
    ax1.set_ylabel("Opinion Score (-1 to 1)", fontsize=12)
    ax1.axhline(0, color="black", linestyle="--", alpha=0.5)

    for i, comm in enumerate(comm_order):
        media_val = df_stats[df_stats["Community"] == comm]["Top Media Type"].values[0]
        if isinstance(media_val, str) and len(media_val) > 10:
            media_val = media_val[:8] + ".."
        ax1.text(i, 1.05, media_val, ha="center", va="bottom", fontsize=9, rotation=45)

    # --- Plot 2: Community Size ---
    sns.barplot(
        data=df_stats,
        x="Community",
        y="Users",
        order=comm_order,
        ax=ax2,
        color="#4c72b0",
    )
    ax2.set_ylabel("Number of Users", fontsize=12)

    for i, comm in enumerate(comm_order):
        n_users = df_stats[df_stats["Community"] == comm]["Users"].values[0]
        ax2.text(
            i,
            n_users + (df_stats["Users"].max() * 0.05),
            str(n_users),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --- Plot 3: Factual Reporting (%) ---
    sns.barplot(
        data=df_stats,
        x="Community",
        y="% High Factual",
        order=comm_order,
        ax=ax3,
        color="#55a868",
    )
    ax3.set_ylabel("% High Factual", fontsize=12)
    ax3.set_ylim(0, 100)

    # --- Plot 4: Credibility (%) ---
    sns.barplot(
        data=df_stats,
        x="Community",
        y="% Mixed Credibility",
        order=comm_order,
        ax=ax4,
        color="#dd8452",
    )
    ax4.set_ylabel("% Mixed Credibility", fontsize=12)
    ax4.set_xlabel("Leiden Community ID", fontsize=14)
    ax4.set_ylim(0, 100)

    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {OUTPUT_PLOT}")
    plt.show()


if __name__ == "__main__":
    main()
