import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import os

from config.paths import TOPICS_PATH

"""
Takes two inputs:
1) A opinion .csv file
2) A network .graphml file. 

Creates two outputs:
1) A .graphml file with global opinion estimates for each user with at least
MIN_URL. The estimate is the average of shared URL opinion scores.
2) A visualization (histogram) of the opinion distribution for the above graph. 
"""

TOPIC_NAME = "#climatechange"

# The corresponding graph for MIN_RP has to exist (see interaction_graph.py)
MIN_RP = 3
MIN_URL = 2

# --- Input ---
GLOBAL_OPINIONS_CSV = (
    TOPICS_PATH / f"{TOPIC_NAME}/eda/opinions_minrp{MIN_RP}_minurl{MIN_URL}.csv"
)

# One can always use a csv where MIN_RP and MIN_URL are equal or lower (more data)
# GLOBAL_OPINIONS_CSV = TOPICS_PATH / f"{TOPIC_NAME}/eda/opinions_minrpx_minurlx.csv"

INPUT_GRAPHML = (
    TOPICS_PATH / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_minrp{MIN_RP}_largecomp.graphml"
)

# --- Output ---
OUTPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_minrp{MIN_RP}_minurl{MIN_URL}.graphml"
)
OUTPUT_PLOT = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/eda/distribution_opinions_minrp{MIN_RP}_minurl{MIN_URL}.png"
)


def process_user_bias_network():
    # 1. Load the pre-calculated Global User Bias Data
    print("Loading Global User Opinion data...")
    df = pd.read_csv(GLOBAL_OPINIONS_CSV)

    # Drop any rows that might be missing an opinion just in case
    df = df.dropna(subset=["estimated_opinion"])

    # Convert user_id to string for safe dictionary matching
    df["user_id"] = df["user_id"].astype(str).str.strip()

    # Create the lookup dictionary directly
    bias_dict = df.set_index("user_id")["estimated_opinion"].to_dict()

    print(f"Loaded global opinion scores for {len(bias_dict)} users.")

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
            # If networkx read it as 371.0, float() -> int() -> str() makes it "371"
            clean_id = str(int(float(raw_id)))
        except (ValueError, TypeError):
            # Fallback just in case there are actual string names
            clean_id = str(raw_id).strip()

        if clean_id in bias_dict:
            matched_score = float(bias_dict[clean_id])
            matched_node_count += 1
        else:
            matched_score = "unknown"

        G.nodes[node]["url_bias_score"] = matched_score

        if matched_score != "unknown":
            user_opinions.append(matched_score)

    print(
        f"Successfully mapped global scores to {matched_node_count} nodes in the graph."
    )

    # Safety check before saving/plotting
    if matched_node_count == 0:
        print(
            "WARNING: No nodes matched. Check if the GraphML contains the correct user IDs."
        )

    os.makedirs(OUTPUT_GRAPHML.parent, exist_ok=True)
    nx.write_graphml(G, OUTPUT_GRAPHML)
    print(f"Updated network saved to {OUTPUT_GRAPHML}")

    # 4. Visualization (Filtered to only show mapped nodes)
    if matched_node_count == 0:
        print("WARNING: No nodes matched. Skipping visualization to prevent errors.")
        return

    print("Generating visualization for mapped nodes only...")
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1)
    ax_graph = fig.add_subplot(gs[0:2, 0])
    ax_hist = fig.add_subplot(gs[2, 0])

    # --- Create a Subgraph of only colored nodes ---
    mapped_nodes = [
        n for n in G.nodes() if G.nodes[n].get("url_bias_score") != "unknown"
    ]
    G_colored = G.subgraph(mapped_nodes)

    print(
        f"Plotting subgraph with {G_colored.number_of_nodes()} mapped nodes and {G_colored.number_of_edges()} remaining edges."
    )

    try:
        # Use existing coordinates if available (e.g. from Gephi ForceAtlas2)
        pos = {
            n: (float(G.nodes[n]["x"]), float(G.nodes[n]["y"]))
            for n in G_colored.nodes()
        }
    except KeyError:
        print("Calculating fast layout for subgraph...")
        # Using a fast layout so it doesn't freeze if Gephi coords are missing
        # pos = nx.spring_layout(G_colored, seed=42, iterations=15)
        pos = nx.random_layout(G_colored, seed=42)

    cmap = plt.get_cmap("coolwarm")

    # --- Custom Normalizer to stretch the "Leftist Bubble" contrast ---
    norm = mcolors.Normalize(vmin=-0.75, vmax=0.25, clip=True)

    # --- FAST NODE DRAWING (Vectorized) ---
    nodelist = list(G_colored.nodes())
    node_color_list = []

    for node in nodelist:
        val = G_colored.nodes[node].get("url_bias_score")
        node_color_list.append(cmap(norm(val)))

    nx.draw_networkx_nodes(
        G_colored,
        pos,
        nodelist=nodelist,
        node_size=30,
        node_color=node_color_list,
        ax=ax_graph,
    )

    # --- FAST EDGE DRAWING (Sampled) ---
    print("Drawing edges (Sampling to prevent Matplotlib freeze)...")
    all_edges = list(G_colored.edges())

    # Cap the maximum number of edges rendered to keep performance snappy
    max_edges_to_draw = 2000
    if len(all_edges) > max_edges_to_draw:
        edges_to_draw = random.sample(all_edges, max_edges_to_draw)
    else:
        edges_to_draw = all_edges

    nx.draw_networkx_edges(
        G_colored,
        pos,
        edgelist=edges_to_draw,
        alpha=0.15,  # Slightly increased opacity since there are fewer lines
        edge_color="grey",
        ax=ax_graph,
    )

    # Update title to reflect the filtering and sampling
    ax_graph.set_title(
        f"{TOPIC_NAME} Global Opinion Network (Only URL-Sharers)\nNodes: {G_colored.number_of_nodes()} | Edges: {G_colored.number_of_edges()} (Showing {len(edges_to_draw)})",
        fontsize=16,
    )
    ax_graph.axis("off")

    # --- Plot Bottom: Histogram ---
    ax_hist.hist(
        user_opinions, bins=40, range=(-1.0, 1.0), edgecolor="black", alpha=0.7
    )

    patches = ax_hist.patches
    for patch in patches:
        x = patch.get_x() + patch.get_width() / 2  # type: ignore

        # Apply the exact same normalizer to the histogram bars
        patch.set_facecolor(cmap(norm(x)))

    ax_hist.set_title("Global User Opinion Distribution", fontsize=16)
    ax_hist.set_xlabel("Opinion (-1 to 1)", fontsize=12)
    ax_hist.set_ylabel("# Users", fontsize=12)
    ax_hist.set_xlim(-1.1, 1.1)
    ax_hist.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Visualization saved to {OUTPUT_PLOT}")


if __name__ == "__main__":
    process_user_bias_network()
