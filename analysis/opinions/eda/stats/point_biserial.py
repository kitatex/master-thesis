import networkx as nx
import numpy as np
from scipy import stats

from config.paths import TOPICS_PATH

# --- Configuration ---
TOPIC_NAME = "immigration"
MIN_RP = 2
ALGORITHM = "leidenmeta"
LEIDEN_RESOLUTION = 0.3

# Point to the final partitioned graph that contains both 'side' and 'url_bias_score'
# Adjust this path if your pipeline saves it differently
INPUT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp{MIN_RP}_part_{ALGORITHM}{LEIDEN_RESOLUTION}.graphml"
)


def run_point_biserial():
    print(f"Loading partitioned graph: {INPUT_GRAPH.name}...")
    G = nx.read_graphml(INPUT_GRAPH)

    sides = []
    opinions = []

    # Extract attributes for nodes that have both
    for node in G.nodes():
        node_attr = G.nodes[node]

        # Check if node has both the partition side and the opinion score
        if "side" in node_attr and "url_bias_score" in node_attr:
            sides.append(int(node_attr["side"]))
            opinions.append(float(node_attr["url_bias_score"]))

    if not sides:
        print("Error: No nodes found with both 'side' and 'url_bias_score' attributes.")
        return

    sides = np.array(sides)
    opinions = np.array(opinions)

    print(f"Successfully extracted data for {len(sides)} users.")
    print("Calculating Point-Biserial Correlation...")

    # Calculate Point-Biserial Correlation
    r_pb, p_value = stats.pointbiserialr(sides, opinions)

    print("\n" + "=" * 40)
    print("POINT-BISERIAL CORRELATION RESULTS")
    print("=" * 40)
    print(f"Correlation Coefficient (r_pb): {r_pb:.4f}")
    print(f"P-value:                        {p_value:.4e}")
    print("=" * 40 + "\n")

    # Brief interpretation output
    if p_value < 0.05:  # type: ignore
        print(
            "Interpretation: The correlation is statistically significant (p < 0.05)."
        )
    else:
        print(
            "Interpretation: The correlation is NOT statistically significant (p >= 0.05)."
        )


if __name__ == "__main__":
    run_point_biserial()
