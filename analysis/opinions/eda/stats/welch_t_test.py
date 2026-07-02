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
INPUT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp{MIN_RP}_part_{ALGORITHM}{LEIDEN_RESOLUTION}.graphml"
)


def run_welch_ttest():
    print(f"Loading partitioned graph: {INPUT_GRAPH.name}...")
    G = nx.read_graphml(INPUT_GRAPH)

    side_0_opinions = []
    side_1_opinions = []

    # Sort opinions into two separate lists based on partition assignment
    for node in G.nodes():
        node_attr = G.nodes[node]

        if "side" in node_attr and "url_bias_score" in node_attr:
            side = int(node_attr["side"])
            score = float(node_attr["url_bias_score"])

            if side == 0:
                side_0_opinions.append(score)
            elif side == 1:
                side_1_opinions.append(score)

    if not side_0_opinions or not side_1_opinions:
        print("Error: Could not find opinion data for both sides.")
        return

    side_0_opinions = np.array(side_0_opinions)
    side_1_opinions = np.array(side_1_opinions)

    print(f"Side 0: {len(side_0_opinions)} users")
    print(f"Side 1: {len(side_1_opinions)} users")
    print("Calculating Welch's t-test...")

    # Calculate Welch's t-test (equal_var=False is what makes it Welch's)
    t_stat, p_value = stats.ttest_ind(side_0_opinions, side_1_opinions, equal_var=False)

    # Descriptive Statistics
    mean_0, std_0 = np.mean(side_0_opinions), np.std(side_0_opinions, ddof=1)
    mean_1, std_1 = np.mean(side_1_opinions), np.std(side_1_opinions, ddof=1)

    print("\n" + "=" * 40)
    print("WELCH'S T-TEST RESULTS")
    print("=" * 40)
    print(f"Side 0 Mean Score: {mean_0:.4f} (Std: {std_0:.4f})")
    print(f"Side 1 Mean Score: {mean_1:.4f} (Std: {std_1:.4f})")
    print("-" * 40)
    print(f"T-statistic:       {t_stat:.4f}")
    print(f"P-value:           {p_value:.4e}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_welch_ttest()
