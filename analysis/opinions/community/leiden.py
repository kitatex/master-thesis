import os
import networkx as nx
from cdlib import algorithms
import warnings

from config.paths import TOPICS_PATH

warnings.filterwarnings("ignore", category=SyntaxWarning)

TOPIC_NAME = "#gamedev"
MIN_RP = 2
MIN_URL = 2

# --- Input (Your fully interpolated graph) ---
INPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_minrp{MIN_RP}_minurl{MIN_URL}_itp.graphml"
)

# --- Output ---
OUTPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/community/{TOPIC_NAME}_minrp{MIN_RP}_minurl{MIN_URL}_itp_cd.graphml"
)


def main():
    print(f"Loading graph file: {INPUT_GRAPHML}")

    # 1. Load the Directed Graph
    G_directed = nx.read_graphml(INPUT_GRAPHML)
    print(f"Graph loaded with {G_directed.number_of_nodes()} nodes.")

    # 2. Convert to Undirected for Community Detection
    # Community algorithms rely on mutual structural density, so we treat edges symmetrically
    print("Converting to undirected for topological analysis...")
    G_undirected = G_directed.to_undirected()

    # 3. Run the Leiden Algorithm
    print("Running Leiden Community Detection...")
    # This returns a NodeClustering object
    leiden_clusters = algorithms.leiden(G_undirected)

    communities = leiden_clusters.communities
    print(f"Leiden algorithm found {len(communities)} distinct structural communities.")

    print("Calculating Modularity Score...")
    # NetworkX handles the math perfectly using your undirected graph and the Leiden clusters
    modularity_score = nx.community.modularity(G_undirected, communities)
    print(f"📊 Modularity Score (Q): {modularity_score:.4f}")
    # ----------------------------

    # 4. Map the Community IDs back to the Directed Graph
    print("Mapping community IDs to nodes...")

    # Create a fast lookup dictionary {node_id: community_id}
    community_mapping = {}
    for cluster_id, node_list in enumerate(communities):
        for node in node_list:
            community_mapping[node] = cluster_id

    # Inject the attribute into the GraphML schema
    for node in G_directed.nodes():
        # We cast the ID to a string so Gephi treats it as a discrete Categorical class, not a continuous math float!
        cluster_id = str(community_mapping.get(node, "unknown"))
        G_directed.nodes[node]["leiden_community"] = cluster_id

    # 5. Save the final Graph
    os.makedirs(OUTPUT_GRAPHML.parent, exist_ok=True)
    nx.write_graphml(G_directed, OUTPUT_GRAPHML)
    print(f"Successfully saved graph with communities to: {OUTPUT_GRAPHML}")


if __name__ == "__main__":
    main()
