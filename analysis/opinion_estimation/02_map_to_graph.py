import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json
import statistics
from collections import Counter
from config.paths import TOPICS_PATH, MBFC_CSV_PATH

# --- Configuration ---
TOPIC_NAME = "war in gaza"
MIN_RP = 2  # has to match csv
MIN_URL = 3  # has to match csv
URL_SOURCE = "global"  # has to match csv; "global" or "discussion"
INTERPOLATE = True  # Set to True to enable continuous label propagation

# --- Input ---
OPINIONS_CSV = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/metadata/opinions_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.csv"
)
URLS_CSV = TOPICS_PATH / f"{TOPIC_NAME}/metadata/extracted_urls_minrp{MIN_RP}.csv"
INPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_minrp{MIN_RP}_largecomp_undir_unweighted.graphml"
)

# --- Output ---
interp_str = "interpolated" if INTERPOLATE else "filtered"
OUTPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/with_opinions/{TOPIC_NAME}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}_{interp_str}.graphml"
)
OUTPUT_PLOT = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/eda/distribution_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}_{interp_str}.png"
)
OUTPUT_STATS = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/metadata/graph_statistics_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}_{interp_str}.json"
)

# Bias mapping from 01_estimate.py
BIAS_MAP = {
    "left": -1.5,
    "left-center": -0.5,
    "neutral": 0.0,
    "right-center": 0.5,
    "right": 1.5,
}


def build_mbfc_lookup(mbfc_csv_path):
    """Loads MBFC and creates a quick lookup dictionary for scores."""
    df = pd.read_csv(mbfc_csv_path)
    if "bias_score" not in df.columns:
        df["bias_score"] = df["bias"].map(BIAS_MAP)
    df = df.dropna(subset=["bias_score"])
    return df.set_index("source")["bias_score"].to_dict()


def propagate_continuous_labels(G, known_dict, max_iter=1000, tol=1e-5):
    """
    Performs iterative label propagation for continuous variables on a network.
    Unlabeled nodes are initialized to the mean of the known nodes and iteratively
    updated to the average of their neighbors until convergence.
    """
    opinions = {}
    known_nodes = set(known_dict.keys())

    mean_known = sum(known_dict.values()) / len(known_dict) if known_dict else 0.0

    # Initialize
    for n in G.nodes():
        if n in known_nodes:
            opinions[n] = known_dict[n]
        else:
            opinions[n] = mean_known

    # Iterative averaging
    for iteration in range(max_iter):
        max_diff = 0
        new_opinions = opinions.copy()

        for n in G.nodes():
            if n not in known_nodes:
                neighbors = list(G.neighbors(n))
                if neighbors:
                    val = sum(opinions[nbr] for nbr in neighbors) / len(neighbors)
                    max_diff = max(max_diff, abs(val - opinions[n]))
                    new_opinions[n] = val

        opinions = new_opinions
        if max_diff < tol:
            print(f"Interpolation converged after {iteration + 1} iterations.")
            break

    return opinions


def process_user_bias_network():
    print(f"Loading {URL_SOURCE} User Opinion data from {OPINIONS_CSV.name}...")
    if not OPINIONS_CSV.exists():
        print(f"Error: Could not find {OPINIONS_CSV}.")
        return

    # 1. Load Pre-calculated Opinions (Seed Nodes)
    df_opinions = pd.read_csv(OPINIONS_CSV)
    df_opinions = df_opinions.dropna(subset=["estimated_opinion"])
    df_opinions["user_id"] = df_opinions["user_id"].astype(str).str.strip()
    bias_dict = df_opinions.set_index("user_id")["estimated_opinion"].to_dict()

    # 2. Load the Network
    print("Loading GraphML network...")
    G = nx.read_graphml(INPUT_GRAPHML)

    # Standardize node IDs
    standardized_nodes = {}
    for node in G.nodes():
        node_attr = G.nodes[node]
        raw_id = node_attr.get("name", node_attr.get("v_name", node))
        try:
            clean_id = str(int(float(raw_id)))
        except (ValueError, TypeError):
            clean_id = str(raw_id).strip()
        standardized_nodes[node] = clean_id

    G = nx.relabel_nodes(G, standardized_nodes)

    # 3. Apply Interpolation or Filtering
    final_opinions = {}
    if INTERPOLATE:
        print("INTERPOLATE is True: Running continuous label propagation...")
        final_opinions = propagate_continuous_labels(G, bias_dict)
        G_processed = G.copy()
    else:
        print("INTERPOLATE is False: Filtering unknown nodes...")
        final_opinions = {n: bias_dict[n] for n in G.nodes() if n in bias_dict}
        G_processed = G.subgraph(final_opinions.keys()).copy()

    # Apply scores to graph
    for node, score in final_opinions.items():
        G_processed.nodes[node]["url_bias_score"] = score
        G_processed.nodes[node]["is_interpolated"] = node not in bias_dict

    # 4. Calculate Statistics for LaTeX Table
    print("\nCalculating summary statistics...")
    mbfc_lookup = build_mbfc_lookup(MBFC_CSV_PATH)

    df_urls = pd.read_csv(URLS_CSV)
    df_urls["user_id"] = df_urls["user_id"].astype(str).str.strip()
    target_column = f"{URL_SOURCE}_urls"

    processed_users = set(G_processed.nodes())
    valid_url_counts = []
    all_matched_urls = []

    for _, row in df_urls.iterrows():
        uid = row["user_id"]
        if uid in processed_users:
            try:
                urls = json.loads(row[target_column])
            except (TypeError, json.JSONDecodeError, KeyError):
                urls = []

            matched_urls = [u for u in urls if u in mbfc_lookup]
            valid_url_counts.append(len(matched_urls))
            all_matched_urls.extend(matched_urls)

    num_nodes = G_processed.number_of_nodes()
    num_edges = G_processed.number_of_edges()
    num_interpolated = sum(
        1 for data in G_processed.nodes(data=True) if data[1].get("is_interpolated")
    )
    pct_interpolated = (num_interpolated / num_nodes * 100) if num_nodes > 0 else 0

    opinion_values = list(final_opinions.values())
    mean_opinion = sum(opinion_values) / len(opinion_values) if opinion_values else 0
    mean_urls_per_user = (
        sum(valid_url_counts) / len(valid_url_counts) if valid_url_counts else 0
    )
    median_urls_per_user = (
        statistics.median(valid_url_counts) if valid_url_counts else 0
    )
    top_10_urls = Counter(all_matched_urls).most_common(10)

    # Compile dictionary for JSON export
    stats_dict = {
        "network_nodes": num_nodes,
        "network_edges": num_edges,
        "interpolated_nodes_count": num_interpolated,
        "interpolated_nodes_percent": round(pct_interpolated, 2),
        "mean_opinion": round(mean_opinion, 4),
        "mean_mbfc_urls_per_user": round(mean_urls_per_user, 2),
        "median_mbfc_urls_per_user": median_urls_per_user,
        "top_10_mbfc_urls": [
            {"domain": domain, "count": count} for domain, count in top_10_urls
        ],
    }

    print("\n" + "=" * 50)
    print("GRAPH STATISTICS (For LaTeX Table)")
    print("=" * 50)
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Interpolated Nodes: {pct_interpolated:.2f}% ({num_interpolated})")
    print(f"Mean Opinion: {mean_opinion:.4f}")
    print(f"Mean MBFC-matched URLs per User: {mean_urls_per_user:.2f}")
    print(f"Median MBFC-matched URLs per User: {median_urls_per_user}")
    print("\nTop 10 Most Common MBFC URLs:")
    for i, (url, count) in enumerate(top_10_urls, 1):
        print(f"  {i}. {url} ({count})")
    print("=" * 50 + "\n")

    # 5. Export Statistics to JSON
    OUTPUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Statistics saved to {OUTPUT_STATS}")

    # 6. Save Graph
    os.makedirs(OUTPUT_GRAPHML.parent, exist_ok=True)
    nx.write_graphml(G_processed, OUTPUT_GRAPHML)
    print(f"Updated network saved to {OUTPUT_GRAPHML}")


if __name__ == "__main__":
    process_user_bias_network()
