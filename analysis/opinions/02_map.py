import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

from config.paths import TOPICS_PATH

"""
Takes two inputs:
1) An opinions .csv file (pre-calculated scores per user)
2) A network .graphml file. 

Creates two outputs:
1) A (potentially reduced) .graphml file mapped with opinion estimates for mapped users.
2) A visualization (histogram) of the opinion distribution. 
"""

TOPIC_NAME = "nuclearpower"

MIN_RP = 1  # has to match csv
MIN_URL = 1  # has to match csv
URL_SOURCE = "global"  # has to match csv; "global" or "discussion"


# --- Input ---
OPINIONS_CSV = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/metadata/opinions_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.csv"
)

INPUT_GRAPHML = (
    TOPICS_PATH / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_minrp{MIN_RP}_allcomp_undir.graphml"
)
INPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_minrp{MIN_RP}_largecomp_undir.graphml"
)

# --- Output ---
OUTPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.graphml"
)
OUTPUT_PLOT = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/eda/distribution_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.png"
)


def process_user_bias_network():
    # 1. Load the pre-calculated User Opinion Data
    print(f"Loading {URL_SOURCE} User Opinion data from {OPINIONS_CSV.name}...")

    if not OPINIONS_CSV.exists():
        print(
            f"Error: Could not find {OPINIONS_CSV}. Please run calculate_bias.py first."
        )
        return

    df = pd.read_csv(OPINIONS_CSV)

    # Drop any rows that might be missing an opinion just in case
    df = df.dropna(subset=["estimated_opinion"])

    # Convert user_id to string for safe dictionary matching
    df["user_id"] = df["user_id"].astype(str).str.strip()

    # Create the lookup dictionary directly
    bias_dict = df.set_index("user_id")["estimated_opinion"].to_dict()

    print(f"Loaded opinion scores for {len(bias_dict)} users.")

    # 2. Load the Network
    print("Loading GraphML network...")
    G = nx.read_graphml(INPUT_GRAPHML)
    print(f"Graph loaded with {G.number_of_nodes()} nodes.")

    # 3. Assign Attributes to Graph
    user_opinions = []
    matched_node_count = 0

    for node in G.nodes():
        node_attr = G.nodes[node]

        # Pull the raw ID from the 'name' or 'v_name' attribute (which igraph used)
        raw_id = node_attr.get("name", node_attr.get("v_name", node))

        # Handle the 'double' float issue from igraph
        try:
            clean_id = str(int(float(raw_id)))
        except (ValueError, TypeError):
            clean_id = str(raw_id).strip()

        if clean_id in bias_dict:
            matched_score = float(bias_dict[clean_id])
            G.nodes[node]["url_bias_score"] = matched_score
            user_opinions.append(matched_score)
            matched_node_count += 1

    print(
        f"Successfully mapped global scores to {matched_node_count} nodes in the graph."
    )

    # Safety check before saving/plotting
    if matched_node_count == 0:
        print(
            "WARNING: No nodes matched. Check if the GraphML contains the correct user IDs."
        )
        return

    # --- 1. CREATE THE SUBGRAPH FIRST ---
    # Filter out anyone who doesn't have a score
    mapped_nodes = [
        n for n in G.nodes() if G.nodes[n].get("url_bias_score") is not None
    ]

    # [FIX]: Add .copy() here. NetworkX views drop dynamic attributes on GraphML export.
    # Copying bakes the attributes securely into the graph object.
    G_colored = G.subgraph(mapped_nodes).copy()

    print(
        f"Filtered subgraph created with {G_colored.number_of_nodes()} mapped nodes and {G_colored.number_of_edges()} remaining edges."
    )

    # --- 2. SAVE THE SUBGRAPH ---
    os.makedirs(OUTPUT_GRAPHML.parent, exist_ok=True)
    nx.write_graphml(G_colored, OUTPUT_GRAPHML)
    print(f"Updated (reduced) network saved to {OUTPUT_GRAPHML}")

    # --- 3. GENERATE VISUALIZATION ---
    print("Generating histogram...")

    # [FIX]: Switched to a single simple subplot instead of the GridSpec
    fig, ax_hist = plt.subplots(figsize=(10, 6))

    # [FIX]: Updated the colormap normalizer to match the -1.5 to 1.5 range
    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=-1.5, vmax=1.5, clip=True)

    # [FIX]: Expanded the histogram range to -1.5 to 1.5
    ax_hist.hist(
        user_opinions, bins=30, range=(-1.5, 1.5), edgecolor="black", alpha=0.7
    )

    patches = ax_hist.patches
    for patch in patches:
        x = patch.get_x() + patch.get_width() / 2  # type: ignore Get center of the bar
        patch.set_facecolor(cmap(norm(x)))

    ax_hist.set_title(
        f"{URL_SOURCE.capitalize()} User Opinion Distribution {TOPIC_NAME} (Min {MIN_URL} URLs)",
        fontsize=16,
    )
    ax_hist.set_xlabel("Opinion (-1.5 to 1.5)", fontsize=12)
    ax_hist.set_ylabel("# Users", fontsize=12)
    ax_hist.set_xlim(-1.6, 1.6)
    ax_hist.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Visualization saved to {OUTPUT_PLOT}")


if __name__ == "__main__":
    process_user_bias_network()
