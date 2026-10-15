import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
import igraph as ig
import leidenalg
from config.paths import TOPICS_PATH, RWC_PATH

TOPIC_NAME = "game development"
MIN_RP = 2
MIN_URL = 3
URL_SOURCE = "global"
LEIDEN_RESOLUTION = 0.3
ALGORITHM = "leidenmeta"
RANDOM_SEED = 42  # Added for reproducibility across runs

INPUT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/with_opinions/{TOPIC_NAME}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.graphml"
)

PARTIT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp{MIN_RP}_part_{ALGORITHM}{LEIDEN_RESOLUTION}.graphml"
)

PARTIT_GRAPH.parent.mkdir(parents=True, exist_ok=True)


def orient_sides_canonically(G, side_0, side_1, attr="url_bias_score"):
    """
    Ensures side_0 ALWAYS represents the partition with the lower mean opinion score
    (e.g., more Left-leaning), eliminating arbitrary eigenvector sign flips.
    """
    scores_0 = [
        float(G.nodes[n][attr]) for n in side_0 if G.nodes[n].get(attr) is not None
    ]
    scores_1 = [
        float(G.nodes[n][attr]) for n in side_1 if G.nodes[n].get(attr) is not None
    ]

    # Fallback: if opinion scores are missing on one side, orient by partition size
    if not scores_0 or not scores_1:
        print(
            "Warning: Missing opinion scores on one side. Orienting by size (side_0 = smaller)."
        )
        if len(side_0) > len(side_1):
            return side_1, side_0
        return side_0, side_1

    mean_0 = np.mean(scores_0)
    mean_1 = np.mean(scores_1)

    if mean_0 > mean_1:
        print(
            f"Swapping partition labels for canonical orientation "
            f"(old Side 0 mean: {mean_0:.4f} > old Side 1 mean: {mean_1:.4f})"
        )
        return side_1, side_0

    return side_0, side_1


def build_membership(side_0, side_1):
    membership = {}
    for node in side_0:
        membership[node] = 0
    for node in side_1:
        membership[node] = 1
    return membership


def spectral_partition_graph(G):
    n_nodes = G.number_of_nodes()
    if n_nodes < 2:
        raise ValueError(
            f"Metagraph only has {n_nodes} community. "
            f"Increase LEIDEN_RESOLUTION (currently {LEIDEN_RESOLUTION})."
        )

    # Guard against disconnected metagraphs (where the 2nd smallest eigenvalue is also 0)
    if not nx.is_connected(G):
        print(
            "Warning: Community metagraph is disconnected! Extracting giant metagraph component..."
        )
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc).copy()
    else:
        G_sub = G

    # Explicitly pass weight="weight" so inter-community edge counts drive the Laplacian cut
    L = nx.normalized_laplacian_matrix(G_sub, weight="weight")
    _, eigenvectors = eigsh(L, k=2, which="SM", maxiter=5000)
    fiedler = eigenvectors[:, 1]

    nodes = list(G_sub.nodes())
    side_0 = set()
    side_1 = set()

    for node, value in zip(nodes, fiedler):
        if value < 0:
            side_0.add(node)
        else:
            side_1.add(node)

    # If there were isolated metagraph nodes, assign them to the smaller side
    if len(G_sub) < len(G):
        remaining = set(G.nodes()) - set(G_sub.nodes())
        for node in remaining:
            if len(side_0) <= len(side_1):
                side_0.add(node)
            else:
                side_1.add(node)

    return side_0, side_1


def leiden_metagraph_partition(G):
    print(f"Running Leiden (resolution={LEIDEN_RESOLUTION}, seed={RANDOM_SEED})...")
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

    g = ig.Graph(n=len(nodes), edges=edges, directed=False)

    result = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=LEIDEN_RESOLUTION,
        seed=RANDOM_SEED,
    )

    communities = []
    for community in result:
        communities.append({nodes[idx] for idx in community})

    print(f"Detected {len(communities)} Leiden communities")
    print("Building weighted community metagraph...")

    community_graph = nx.Graph()
    for comm_id in range(len(communities)):
        community_graph.add_node(comm_id)

    node_to_community = {}
    for comm_id, community in enumerate(communities):
        for node in community:
            node_to_community[node] = comm_id

    for u, v in G.edges():
        cu = node_to_community[u]
        cv = node_to_community[v]
        if cu == cv:
            continue

        # Support weighted input graphs if edge weights exist, otherwise count edges as 1.0
        edge_w = float(G[u][v].get("weight", 1.0))
        if community_graph.has_edge(cu, cv):
            community_graph[cu][cv]["weight"] += edge_w
        else:
            community_graph.add_edge(cu, cv, weight=edge_w)

    print(
        f"Community metagraph: "
        f"{community_graph.number_of_nodes()} nodes, "
        f"{community_graph.number_of_edges()} weighted edges"
    )

    print("Spectral partition of community metagraph...")
    meta_side_0, meta_side_1 = spectral_partition_graph(community_graph)

    side_0 = set()
    side_1 = set()
    for comm_id in meta_side_0:
        side_0.update(communities[comm_id])
    for comm_id in meta_side_1:
        side_1.update(communities[comm_id])

    # Canonically orient so Side 0 always has the lower mean opinion score
    side_0, side_1 = orient_sides_canonically(G, side_0, side_1, attr="url_bias_score")

    return side_0, side_1


if __name__ == "__main__":
    print("Loading graph...")
    graph_nx = nx.read_graphml(INPUT_GRAPH)
    graph_nx = graph_nx.to_undirected()
    print(
        f"Original graph: "
        f"{graph_nx.number_of_nodes()} nodes, "
        f"{graph_nx.number_of_edges()} edges"
    )

    # Giant component
    components = list(nx.connected_components(graph_nx))
    giant_component_nodes = max(components, key=len)
    graph_nx = graph_nx.subgraph(giant_component_nodes).copy()
    print(
        f"Giant component: "
        f"{graph_nx.number_of_nodes()} nodes, "
        f"{graph_nx.number_of_edges()} edges"
    )

    # Partition (now canonically oriented!)
    side_0, side_1 = leiden_metagraph_partition(graph_nx)

    # Statistics
    cut_edges = nx.cut_size(graph_nx, side_0, side_1)
    n = graph_nx.number_of_nodes()

    scores_0 = [
        float(graph_nx.nodes[u]["url_bias_score"])
        for u in side_0
        if "url_bias_score" in graph_nx.nodes[u]
    ]
    scores_1 = [
        float(graph_nx.nodes[u]["url_bias_score"])
        for u in side_1
        if "url_bias_score" in graph_nx.nodes[u]
    ]

    print("\nPartition statistics")
    print("--------------------")
    print(
        f"Side 0 (Lower/Left):  {len(side_0)} nodes (Mean opinion: {np.mean(scores_0):.4f})"
    )
    print(
        f"Side 1 (Higher/Right): {len(side_1)} nodes (Mean opinion: {np.mean(scores_1):.4f})"
    )
    print(f"Side 0 Ratio: {len(side_0) / n:.2%}")
    print(f"Cut edges: {cut_edges}")

    # Save
    membership = build_membership(side_0, side_1)
    nx.set_node_attributes(graph_nx, membership, "side")
    nx.set_node_attributes(
        graph_nx, {node: ALGORITHM for node in graph_nx.nodes()}, "algorithm"
    )

    nx.write_graphml(graph_nx, str(PARTIT_GRAPH))
    print(f"\nPartitioned GraphML saved to:\n{PARTIT_GRAPH}")
