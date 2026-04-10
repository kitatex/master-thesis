import random
import igraph as ig

from data_processing.config.paths import TOPICS_PATH


TOPIC_NAME = "#aiethics"  # e.g., #climatecrisis


PATH_PARTITIONED_GRAPH = TOPICS_PATH / f"{TOPIC_NAME}/graph/network_partitioned.graphml"


def calculate_rwc(g: ig.Graph, side_0: list, side_1: list, sample_percent: float = 0.1):
    # Sample nodes to act as the "user nodes" (the nodes we check if we land on)
    k_0 = int(len(side_0) * sample_percent)
    k_1 = int(len(side_1) * sample_percent)

    # Fast lookup sets
    target_nodes_0 = set(random.sample(side_0, k_0))
    target_nodes_1 = set(random.sample(side_1, k_1))

    def simulate_walk(start_node, target_set_same, target_set_other):
        # We start the walk and look for the FIRST target node we hit
        current_node = start_node
        steps = 0
        max_steps = g.ecount() * 2  # Prevent infinite loops in disconnected components

        while steps < max_steps:
            neighbors = g.neighbors(current_node)
            if not neighbors:
                return None  # Dead end

            # Step to a random neighbor
            current_node = random.choice(neighbors)
            steps += 1

            # Check if we hit a target (excluding the start node itself)
            if current_node in target_set_same and current_node != start_node:
                return "same"
            if current_node in target_set_other:
                return "other"

        return None

    # Track outcomes: [Started in 0 -> Ended in 0, Started in 0 -> Ended in 1]
    results_0 = {"same": 0, "other": 0}
    for node in target_nodes_0:
        res = simulate_walk(node, target_nodes_0, target_nodes_1)
        if res:
            results_0[res] += 1

    results_1 = {"same": 0, "other": 0}
    for node in target_nodes_1:
        res = simulate_walk(node, target_nodes_1, target_nodes_0)
        if res:
            results_1[res] += 1

    # Calculate Probabilities
    # P_XX = Probability of starting in X and ending in X
    P_00 = results_0["same"] / sum(results_0.values()) if sum(results_0.values()) else 0
    P_01 = (
        results_0["other"] / sum(results_0.values()) if sum(results_0.values()) else 0
    )

    P_11 = results_1["same"] / sum(results_1.values()) if sum(results_1.values()) else 0
    P_10 = (
        results_1["other"] / sum(results_1.values()) if sum(results_1.values()) else 0
    )

    # RWC Score = P(0->0) * P(1->1) - P(0->1) * P(1->0)
    rwc_score = (P_00 * P_11) - (P_01 * P_10)

    return rwc_score, P_00, P_11


if __name__ == "__main__":
    print(f"Loading graph from {PATH_PARTITIONED_GRAPH}...")
    g = ig.Graph.Read_GraphML(str(PATH_PARTITIONED_GRAPH))

    # Extract the two sides based on the saved 'side' attribute
    # Note: GraphML sometimes saves numeric attributes as floats/strings, so cast to int just in case
    side_0 = [v.index for v in g.vs if int(v["side"]) == 0]
    side_1 = [v.index for v in g.vs if int(v["side"]) == 1]

    print(f"Side 0 size: {len(side_0)}")
    print(f"Side 1 size: {len(side_1)}")

    print("Calculating RWC Score...")
    rwc, p00, p11 = calculate_rwc(g, side_0, side_1, sample_percent=0.1)

    print(f"P(0->0): {p00:.4f}")
    print(f"P(1->1): {p11:.4f}")
    print(f"Final Controversy Score (RWC): {rwc:.4f}")
