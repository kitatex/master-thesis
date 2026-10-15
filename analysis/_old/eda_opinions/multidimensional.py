import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import itertools
import json
import ast
import os

from config.paths import TOPICS_PATH, MBFC_CSV_PATH

# --- Configuration & Paths ---
TOPIC_NAME = "#gaza"
MIN_RP = 2
MIN_URL = 2

OPINIONS_CSV = TOPICS_PATH / f"{TOPIC_NAME}/eda/opinions_minrp2_minurl3.csv"

MBFC_CSV = MBFC_CSV_PATH

OUTPUT_DIR = TOPICS_PATH / f"{TOPIC_NAME}/eda/multidimensional/"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_parse_urls(url_string):
    """Safely parse the messy CSV string representation of the URL list."""
    try:
        # Sometimes pandas reads it with double quotes escaped
        clean_str = url_string.replace('""', '"')
        return json.loads(clean_str)
    except:
        try:
            return ast.literal_eval(url_string)
        except:
            return []


def main():
    print("1. Loading and Preprocessing Data...")

    # Load Data
    users_df = pd.read_csv(OPINIONS_CSV)
    mbfc_df = pd.read_csv(MBFC_CSV)

    # Parse the URL strings into actual Python lists
    users_df["url_list"] = users_df["urls"].apply(safe_parse_urls)

    # "Explode" the dataframe so each user-url pair gets its own row
    shares_df = users_df[["user_id", "estimated_opinion", "url_list"]].explode(
        "url_list"
    )
    shares_df = shares_df.rename(columns={"url_list": "source"})

    # Merge with MBFC metadata
    merged_df = shares_df.merge(mbfc_df, on="source", how="inner")

    print(f"Matched {len(merged_df)} total URL shares to the MBFC database.")

    # ==========================================
    # METHOD 1: MULTI-DIMENSIONAL USER PROFILING
    # ==========================================
    print("\n2. Generating User-Level Correlation Matrix...")

    # Map categorical MBFC columns to ordinal numeric values for correlation math
    credibility_map = {
        "low credibility": 1,
        "mixed credibility": 2,
        "mostly credible": 3,
        "high credibility": 4,
    }
    factual_map = {
        "very low": 1,
        "low": 2,
        "mixed": 3,
        "mostly factual": 4,
        "high": 5,
        "very high": 6,
    }

    merged_df["credibility_score"] = (
        merged_df["mbfc_credibility_rating"].str.lower().map(credibility_map)
    )
    merged_df["factual_score"] = (
        merged_df["factual_reporting"].str.lower().map(factual_map)
    )

    # Aggregate back to the User level to create a "Diet Profile"
    user_profiles = (
        merged_df.groupby("user_id")
        .agg(
            estimated_opinion=(
                "estimated_opinion",
                "first",
            ),  # Same for all rows of a user
            avg_credibility=("credibility_score", "mean"),
            avg_factual=("factual_score", "mean"),
            unique_sources=("source", "nunique"),
            total_shares=("source", "count"),
        )
        .reset_index()
    )

    # Calculate Polarization (Absolute distance from 0 / Neutral)
    # This checks if EXTREMISM (left or right) correlates with credibility
    user_profiles["polarization_strength"] = user_profiles["estimated_opinion"].abs()

    # Plot Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr_cols = [
        "estimated_opinion",
        "polarization_strength",
        "avg_credibility",
        "avg_factual",
        "unique_sources",
    ]
    sns.heatmap(
        user_profiles[corr_cols].corr(),
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
    )
    plt.title(f"User Diet Correlations ({TOPIC_NAME})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/user_correlation_heatmap.png", dpi=300)
    plt.close()
    print(f" -> Saved correlation heatmap to {OUTPUT_DIR}")

    # ==========================================
    # METHOD 2: SOURCE CO-OCCURRENCE NETWORK
    # ==========================================
    print("\n3. Building Source Co-occurrence Network...")

    # We want to see which sources are frequently shared by the SAME user.
    # Group by user_id and get a list of unique sources they shared
    user_source_lists = (
        merged_df.groupby("user_id")["source"].apply(lambda x: list(set(x))).tolist()
    )

    # Count co-occurrences
    co_occurrence = {}
    for source_list in user_source_lists:
        # Create all unique pairs of sources shared by this user
        for pair in itertools.combinations(sorted(source_list), 2):
            if pair not in co_occurrence:
                co_occurrence[pair] = 0
            co_occurrence[pair] += 1

    # Convert to a NetworkX Graph
    G = nx.Graph()
    for (source_a, source_b), weight in co_occurrence.items():
        if (
            weight >= 5
        ):  # MINIMUM THRESHOLD: Sources must be co-shared by at least 5 users to form an edge
            G.add_edge(source_a, source_b, weight=weight)

    # Add MBFC attributes to the nodes for visualization in Gephi later
    mbfc_dict = mbfc_df.set_index("source").to_dict("index")
    for node in G.nodes():
        if node in mbfc_dict:
            for key, val in mbfc_dict[node].items():
                G.nodes[node][str(key)] = str(
                    val
                )  # Convert to string to avoid Gephi XML errors

    # Extract Giant Component to remove noise
    if len(G) > 0:
        giant_comp = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        print(
            f" -> Giant component has {giant_comp.number_of_nodes()} sources and {giant_comp.number_of_edges()} edges."
        )

        # Save for Gephi
        nx.write_graphml(giant_comp, f"{OUTPUT_DIR}/source_cooccurrence.graphml")
        print(f" -> Saved Source Network to {OUTPUT_DIR}/source_cooccurrence.graphml")
    else:
        print(" -> Not enough co-occurrences to build a network.")

    # ==========================================
    # METHOD 3: FACTUAL REPORTING BY IDEOLOGY
    # ==========================================
    print("\n4. Plotting Ideology vs. Factual Reporting Distribution...")

    plt.figure(figsize=(12, 6))

    # Sort the factual reporting categories logically
    order = ["very low", "low", "mixed", "mostly factual", "high", "very high"]

    # We plot every individual share to see the distribution of ideology for each fact-level
    sns.violinplot(
        data=merged_df,
        x="factual_reporting",
        y="estimated_opinion",
        order=order,
        palette="coolwarm",
        inner="quartile",
    )
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.title(
        f"Ideological Distribution of URLs by Factual Reporting Level ({TOPIC_NAME})"
    )
    plt.xlabel("MBFC Factual Reporting Rating")
    plt.ylabel("User's Estimated Opinion (-1 Left to +1 Right)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ideology_vs_factual_violin.png", dpi=300)
    plt.close()
    print(f" -> Saved Violin Plot to {OUTPUT_DIR}")

    print("\nEDA Pipeline Complete!")


if __name__ == "__main__":
    main()
