import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
import igraph as ig
import leidenalg
from config.paths import TOPICS_PATH

TOPIC_NAME = "0 pokemon"
MIN_RP = 2
MIN_URL = 3
URL_SOURCE = "global"

INTERPOLATED = True
LEIDEN_RESOLUTION = 1  # default
ALGORITHM = "leidenmeta"
RANDOM_SEED = 123


def get_opinion_graph_path(
    topic: str,
    url_source: str = "global",
    min_rp: int = 2,
    min_url: int = 3,
    interpolated: bool = True,
):
    """Single source of truth for loading graphs from 02_map_to_graph.py."""
    base_dir = TOPICS_PATH / f"{topic}/graph/with_opinions"
    suffix = "_interpolated" if interpolated else "_filtered"
    primary = (
        base_dir / f"{topic}_{url_source}_minrp{min_rp}_minurl{min_url}{suffix}.graphml"
    )
    if primary.exists():
        return primary
    # Fallback to legacy filename without _interpolated/_filtered
    print("WARNING!")
    return base_dir / f"{topic}_{url_source}_minrp{min_rp}_minurl{min_url}.graphml"


def extract_giant_component(G: nx.Graph) -> nx.Graph:
    """Converts to undirected and extracts the giant connected component."""
    G_undir = G.to_undirected()
    components = list(nx.connected_components(G_undir))
    if not components:
        return G_undir
    giant_nodes = max(components, key=len)
    return G_undir.subgraph(giant_nodes).copy()


def orient_sides_canonically(
    G: nx.Graph, side_0: set, side_1: set, attr: str = "url_bias_score"
):
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

    if not scores_0 or not scores_1:
        if len(side_0) > len(side_1):
            return side_1, side_0
        return side_0, side_1

    if np.mean(scores_0) > np.mean(scores_1):
        return side_1, side_0

    return side_0, side_1


def build_membership(side_0: set, side_1: set) -> dict:
    membership = {node: 0 for node in side_0}
    membership.update({node: 1 for node in side_1})
    return membership


def build_community_metagraph(
    n_communities: int, membership: list, edges: list
) -> nx.Graph:
    """
    Builds the weighted Leiden community metagraph from integer edge tuples (u_idx, v_idx)
    or weighted tuples (u_idx, v_idx, weight).
    """
    community_graph = nx.Graph()
    community_graph.add_nodes_from(range(n_communities))

    for edge in edges:
        u_idx, v_idx = edge[0], edge[1]
        w = edge[2] if len(edge) > 2 else 1.0
        cu = membership[u_idx]
        cv = membership[v_idx]
        if cu == cv:
            continue
        if community_graph.has_edge(cu, cv):
            community_graph[cu][cv]["weight"] += w
        else:
            community_graph.add_edge(cu, cv, weight=w)

    return community_graph


def spectral_partition_graph(G: nx.Graph):
    """Performs normalized Laplacian spectral bisection on the community metagraph."""
    n_nodes = G.number_of_nodes()
    if n_nodes < 2:
        raise ValueError(
            f"Metagraph only has {n_nodes} community. Increase resolution."
        )

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc).copy()
    else:
        G_sub = G

    if G_sub.number_of_nodes() < 2:
        raise ValueError("Giant component of metagraph has < 2 nodes.")

    L = nx.normalized_laplacian_matrix(G_sub, weight="weight")
    _, eigenvectors = eigsh(L, k=2, which="SM", maxiter=5000)
    fiedler = eigenvectors[:, 1]

    side_0 = set()
    side_1 = set()
    for node, value in zip(G_sub.nodes(), fiedler):
        if value < 0:
            side_0.add(node)
        else:
            side_1.add(node)

    # Assign any isolated metagraph communities to the smaller side
    if len(G_sub) < len(G):
        for node in set(G.nodes()) - set(G_sub.nodes()):
            if len(side_0) <= len(side_1):
                side_0.add(node)
            else:
                side_1.add(node)

    if not side_0 or not side_1:
        raise ValueError("Spectral bisection resulted in an empty partition side.")

    return side_0, side_1


def leiden_metagraph_partition(
    G: nx.Graph,
    resolution: float = LEIDEN_RESOLUTION,
    seed: int = RANDOM_SEED,
    orient_by_opinion: bool = True,
    verbose: bool = True,
):
    """
    Full Leiden + Spectral Metagraph Partitioning pipeline on a NetworkX graph G.
    """
    if verbose:
        print(f"Running Leiden (resolution={resolution}, seed={seed})...")

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_idx = [
        (node_to_idx[u], node_to_idx[v], float(G[u][v].get("weight", 1.0)))
        for u, v in G.edges()
    ]

    g_ig = ig.Graph(
        n=len(nodes),
        edges=[(u, v) for u, v, _ in edges_idx],
        directed=False,
    )

    result = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )

    n_comms = len(result)
    if verbose:
        print(f"Detected {n_comms} Leiden communities")

    community_graph = build_community_metagraph(n_comms, result.membership, edges_idx)
    if verbose:
        print(
            f"Community metagraph: {community_graph.number_of_nodes()} nodes, "
            f"{community_graph.number_of_edges()} weighted edges"
        )

    meta_side_0, meta_side_1 = spectral_partition_graph(community_graph)

    side_0 = {nodes[idx] for comm_id in meta_side_0 for idx in result[comm_id]}
    side_1 = {nodes[idx] for comm_id in meta_side_1 for idx in result[comm_id]}

    if orient_by_opinion:
        side_0, side_1 = orient_sides_canonically(
            G, side_0, side_1, attr="url_bias_score"
        )

    return side_0, side_1


if __name__ == "__main__":
    input_graph_path = get_opinion_graph_path(
        TOPIC_NAME, URL_SOURCE, MIN_RP, MIN_URL, INTERPOLATED
    )
    partit_graph_path = (
        TOPICS_PATH
        / f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp{MIN_RP}_part_{ALGORITHM}{LEIDEN_RESOLUTION}.graphml"
    )
    partit_graph_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph from: {input_graph_path}...")
    raw_graph = nx.read_graphml(input_graph_path)
    graph_nx = extract_giant_component(raw_graph)
    print(
        f"Giant component: {graph_nx.number_of_nodes()} nodes, "
        f"{graph_nx.number_of_edges()} edges"
    )

    side_0, side_1 = leiden_metagraph_partition(
        graph_nx, resolution=LEIDEN_RESOLUTION, seed=RANDOM_SEED, verbose=True
    )

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

    membership = build_membership(side_0, side_1)
    nx.set_node_attributes(graph_nx, membership, "side")
    nx.set_node_attributes(
        graph_nx, {node: ALGORITHM for node in graph_nx.nodes()}, "algorithm"
    )

    nx.write_graphml(graph_nx, str(partit_graph_path))
    print(f"\nPartitioned GraphML saved to:\n{partit_graph_path}")
