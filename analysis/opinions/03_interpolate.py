import os
import numpy as np
import networkx as nx

from config.paths import TOPICS_PATH

TOPIC_NAME = "#gamedev"
MIN_RP = 2
MIN_URL = 2

# --- Input ---
INPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_minrp{MIN_RP}_minurl{MIN_URL}.graphml"
)

# --- Output ---
OUTPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_minrp{MIN_RP}_minurl{MIN_URL}_itp.graphml"
)


def calculate_opinion_diffusion(G_undirected, max_iter=500, tolerance=1e-5):
    """
    Runs the DeGroot diffusion math over an undirected topology.
    Returns a dictionary mapping {node_id: final_interpolated_score}.
    """
    opinions = {}
    anchors = set()
    unknowns = set()

    # 1. Identify anchors vs unknowns
    for node in G_undirected.nodes():
        val = G_undirected.nodes[node].get("url_bias_score", "unknown")
        if val != "unknown" and val is not None:
            opinions[node] = float(val)
            anchors.add(node)
        else:
            unknowns.add(node)

    if not anchors:
        print("ERROR: No anchor nodes found with valid 'url_bias_score'.")
        return opinions

    # 2. Initialize unknown nodes
    global_mean = np.mean([opinions[n] for n in anchors])
    for node in unknowns:
        opinions[node] = global_mean

    print(
        f"Starting relaxation over {len(unknowns)} unknown nodes (Anchors: {len(anchors)})..."
    )

    # 3. Iterative Jacobi relaxation loop
    for i in range(max_iter):
        old_opinions = opinions.copy()
        max_diff = 0.0

        for node in unknowns:
            neighbors = list(G_undirected.neighbors(node))
            if not neighbors:
                continue  # Skip isolated nodes

            # Compute the average opinion of all structural neighbors
            neighbor_opinions = [old_opinions[nbr] for nbr in neighbors]
            new_opinion = sum(neighbor_opinions) / len(neighbor_opinions)

            opinions[node] = new_opinion

            # Track convergence tracking
            diff = abs(new_opinion - old_opinions[node])
            if diff > max_diff:
                max_diff = diff

        print(f" > Iteration {i + 1:02d}: Max delta error = {max_diff:.6f}")

        # Stop early if the network settles
        if max_diff < tolerance:
            print(f"Converged early at iteration {i + 1}!")
            break

    return opinions


def main():
    print(f"Loading directed graph file: {INPUT_GRAPHML}")
    if not INPUT_GRAPHML.exists():
        print(f"File not found: {INPUT_GRAPHML}")
        return

    # 1. Load the original Directed Graph (This acts as our pristine storage container)
    G_directed = nx.read_graphml(INPUT_GRAPHML)
    print(
        f"Graph loaded with {G_directed.number_of_nodes()} nodes and {G_directed.number_of_edges()} edges."
    )

    # 2. Create an Undirected copy purely to calculate symmetric homophily
    print("Creating undirected structural copy for math calculations...")
    G_undirected = G_directed.to_undirected()

    # 3. Calculate the diffused scores (Returns a simple dictionary)
    computed_opinions = calculate_opinion_diffusion(G_undirected)

    # 4. Target Injection: Write the scores directly into the Directed Graph
    print("Injecting computed scores back into the Directed network schema...")
    for node in G_directed.nodes():
        # Retrieve the computed score (fallback to 0.0 if a node was completely isolated)
        final_score = computed_opinions.get(node, 0.0)
        G_directed.nodes[node]["interpolated_bias_score"] = float(final_score)

    # 5. Save the final Directed Graph
    os.makedirs(OUTPUT_GRAPHML.parent, exist_ok=True)
    nx.write_graphml(G_directed, OUTPUT_GRAPHML)
    print(f"Successfully saved fully interpolated DIRECTED graph to: {OUTPUT_GRAPHML}")


if __name__ == "__main__":
    main()
