import networkx as nx
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

INPUT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/with_opinions/{TOPIC_NAME}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.graphml"
)
PARTIT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp{MIN_RP}_part_{ALGORITHM}{LEIDEN_RESOLUTION}.graphml"
)

# Paper reproduction
# GARIMELLA_GRAPH = "retweet_graph_sxsw_threshold_largest_CC_undir_gcc.graphml"
# INPUT_GRAPH = RWC_PATH / f"graphs_graphml_selection/{GARIMELLA_GRAPH}"
# PARTIT_GRAPH = RWC_PATH / f"graphs_graphml_selection/partit/{GARIMELLA_GRAPH}"


PARTIT_GRAPH.parent.mkdir(parents=True, exist_ok=True)


def build_membership(side_0, side_1):
    membership = {}

    for node in side_0:
        membership[node] = 0

    for node in side_1:
        membership[node] = 1

    return membership


def spectral_partition_graph(G):
    L = nx.normalized_laplacian_matrix(G)

    _, eigenvectors = eigsh(L, k=2, which="SM")

    fiedler = eigenvectors[:, 1]

    nodes = list(G.nodes())

    side_0 = set()
    side_1 = set()

    for node, value in zip(nodes, fiedler):
        if value < 0:
            side_0.add(node)
        else:
            side_1.add(node)

    return side_0, side_1


def leiden_metagraph_partition(G):
    print("Running Leiden...")

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

    g = ig.Graph(n=len(nodes), edges=edges, directed=False)

    result = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=LEIDEN_RESOLUTION,
    )

    communities = []

    for community in result:
        communities.append({nodes[idx] for idx in community})

    print(f"Detected {len(communities)} Leiden communities")

    print("Building community graph...")

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

        if community_graph.has_edge(cu, cv):
            community_graph[cu][cv]["weight"] += 1
        else:
            community_graph.add_edge(
                cu,
                cv,
                weight=1,
            )

    print(
        f"Community graph: "
        f"{community_graph.number_of_nodes()} nodes, "
        f"{community_graph.number_of_edges()} edges"
    )

    print("Spectral partition of community graph...")

    meta_side_0, meta_side_1 = spectral_partition_graph(community_graph)

    side_0 = set()
    side_1 = set()

    for comm_id in meta_side_0:
        side_0.update(communities[comm_id])

    for comm_id in meta_side_1:
        side_1.update(communities[comm_id])

    return side_0, side_1


# ----------------------------------------------------
# Load graph
# ----------------------------------------------------

print("Loading graph...")

graph_nx = nx.read_graphml(INPUT_GRAPH)
graph_nx = graph_nx.to_undirected()

print(
    f"Original graph: "
    f"{graph_nx.number_of_nodes()} nodes, "
    f"{graph_nx.number_of_edges()} edges"
)

# ----------------------------------------------------
# Giant component
# ----------------------------------------------------

components = list(nx.connected_components(graph_nx))

giant_component_nodes = max(components, key=len)

graph_nx = graph_nx.subgraph(giant_component_nodes).copy()

print(
    f"Giant component: "
    f"{graph_nx.number_of_nodes()} nodes, "
    f"{graph_nx.number_of_edges()} edges"
)

# ----------------------------------------------------
# Partition
# ----------------------------------------------------

side_0, side_1 = leiden_metagraph_partition(graph_nx)

# ----------------------------------------------------
# Statistics
# ----------------------------------------------------

cut_edges = nx.cut_size(graph_nx, side_0, side_1)

n = graph_nx.number_of_nodes()

print("\nPartition statistics")
print("--------------------")
print(f"Side 0: {len(side_0)}")
print(f"Side 1: {len(side_1)}")
print(f"Ratio: {len(side_0) / n:.2%}")
print(f"Cut edges: {cut_edges}")

# ----------------------------------------------------
# Save
# ----------------------------------------------------

membership = build_membership(side_0, side_1)

nx.set_node_attributes(graph_nx, membership, "side")

nx.set_node_attributes(
    graph_nx, {node: ALGORITHM for node in graph_nx.nodes()}, "algorithm"
)

nx.write_graphml(graph_nx, str(PARTIT_GRAPH))

print(f"\nPartitioned GraphML saved to:\n{PARTIT_GRAPH}")
