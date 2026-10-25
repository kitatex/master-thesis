import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from config.paths import TOPICS_PATH

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
TOPIC_GROUPS = {
    "Political": [
        "capitalism",
        "climate change",
        "covid-19",
        "immigration",
        "gaza",
        "ukraine",
    ],
    "Non-Controversial (Control)": [
        "0 basketball",
        "0 bg3",
        "0 coffee",
        "0 game development",
        "0 photography",
        "0 pokemon",
    ],
}

MIN_RP = 2
LEIDEN_RESOLUTION = 1.0
ALGORITHM = "leidenmeta"


def plot_topic_distribution(topic: str, group: str):
    """
    Loads the partitioned graph for a single topic, filters for ground-truth seed users,
    and generates/saves the opinion density distribution plot.
    """
    # Formatting to match standard Leiden float naming (e.g., leidenmeta1.0)
    res_str = (
        int(LEIDEN_RESOLUTION) if LEIDEN_RESOLUTION.is_integer() else LEIDEN_RESOLUTION
    )
    file_name = f"{topic}_minrp{MIN_RP}_part_{ALGORITHM}{res_str}.graphml"

    input_graph = TOPICS_PATH / topic / "graph" / "partition" / file_name
    output_plot = (
        TOPICS_PATH
        / topic
        / "eda"
        / f"{topic}_sides_distribution_{ALGORITHM}{res_str}.png"
    )

    if not input_graph.exists():
        print(f"[WARNING] Graph not found for '{topic}': {input_graph}")
        return

    G = nx.read_graphml(input_graph)

    side0 = []
    side1 = []

    for node, attrs in G.nodes(data=True):
        opinion = attrs.get("url_bias_score")
        side = attrs.get("side")

        # Safely parse the interpolated flag to ensure we only plot ground-truth seeds
        is_interpolated = attrs.get("is_interpolated", False)
        if isinstance(is_interpolated, str):
            is_interpolated = is_interpolated.strip().lower() == "true"
        else:
            is_interpolated = bool(is_interpolated)

        if opinion is None or side is None or is_interpolated:
            continue

        opinion = float(opinion)
        side = int(side)

        if side == 0:
            side0.append(opinion)
        elif side == 1:
            side1.append(opinion)

    if not side0 and not side1:
        print(f"Skipping '{topic}': No valid seed users found.")
        return

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Fallback to standard bins if a side has < 2 users to prevent numpy 'fd' estimator crashes
    bins_0 = "fd" if len(side0) > 1 else 10
    bins_1 = "fd" if len(side1) > 1 else 10

    if side0:
        plt.hist(
            side0,
            bins=bins_0,
            density=True,
            alpha=0.5,
            color="#E66100",
            label=f"Side 0 (n={len(side0)})",
        )

    if side1:
        plt.hist(
            side1,
            bins=bins_1,
            density=True,
            alpha=0.5,
            color="#5D3A9B",
            label=f"Side 1 (n={len(side1)})",
        )

    # Clean up topic name for the title (remove "0 " and capitalize)
    clean_title = topic.replace("0 ", "").title()

    plt.xlabel("Estimated Opinion (Seed Users Only)", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    plt.title(
        f"{clean_title} - Opinion Distribution",
        pad=10,
        fontweight="bold",
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(
        -1.6, 1.6
    )  # Standardize x-axis across all plots for easy visual comparison

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)

    # CRITICAL: Close the figure to free memory and prevent plots from overlapping in the loop
    plt.close()

    print(f"Saved: {output_plot.name} (Side 0: {len(side0)}, Side 1: {len(side1)})")


def main():
    print(
        f"Starting batch distribution plotting for {sum(len(v) for v in TOPIC_GROUPS.values())} topics...\n"
    )

    for group, topics in TOPIC_GROUPS.items():
        for topic in topics:
            plot_topic_distribution(topic, group)

    print("\nBatch plotting complete!")


if __name__ == "__main__":
    main()
