import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
from scipy.sparse.linalg import eigsh
import community as community_louvain
import igraph as ig
import leidenalg

from config.paths import TOPICS_PATH

TOPIC_NAME = "#gaza"
MIN_RP = 2
MIN_URL = 3
URL_SOURCE = "global"

ALGORITHM = "spectral"  # kern, spectral, louvain, leiden

INPUT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/with_opinions/{TOPIC_NAME}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.graphml"
)

# #gaza #gaza_minrp2_largecomp_undir_unweighted

INPUT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_minrp{MIN_RP}_largecomp_undir_unweighted.graphml"
)


PARTIT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp{MIN_RP}_part_{ALGORITHM}.graphml"
)

PARTIT_GRAPH.parent.mkdir(parents=True, exist_ok=True)


def build_membership(side_0, side_1):
    membership = {}

    for node in side_0:
        membership[node] = 0

    for node in side_1:
        membership[node] = 1

    return membership


def spectral_partition(G):
    print("Computing normalized Laplacian...")

    L = nx.normalized_laplacian_matrix(G)

    print("Computing Fiedler vector...")

    eigenvalues, eigenvectors = eigsh(L, k=2, which="SM")

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


def louvain_partition(G):
    print("Running Louvain...")

    partition = community_louvain.best_partition(G)

    communities = {}

    for node, comm in partition.items():
        communities.setdefault(comm, set()).add(node)

    communities = sorted(communities.values(), key=len, reverse=True)

    print(f"Detected {len(communities)} communities")

    side_0 = communities[0]

    side_1 = set()

    for community in communities[1:]:
        side_1.update(community)

    return side_0, side_1


def leiden_partition(G):
    print("Running Leiden...")

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

    g = ig.Graph(n=len(nodes), edges=edges, directed=False)

    result = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

    communities = []

    for community in result:
        communities.append({nodes[idx] for idx in community})

    communities = sorted(communities, key=len, reverse=True)

    print(f"Detected {len(communities)} communities")

    side_0 = communities[0]

    side_1 = set()

    for community in communities[1:]:
        side_1.update(community)

    return side_0, side_1


# ------------------------------------------------------------------
# Load graph
# ------------------------------------------------------------------

print("Loading graph...")

graph_nx = nx.read_graphml(INPUT_GRAPH)
graph_nx = graph_nx.to_undirected()

print(
    f"Original graph: "
    f"{graph_nx.number_of_nodes()} nodes, "
    f"{graph_nx.number_of_edges()} edges"
)

# ------------------------------------------------------------------
# Giant component
# ------------------------------------------------------------------

components = list(nx.connected_components(graph_nx))
giant_component_nodes = max(components, key=len)

graph_nx = graph_nx.subgraph(giant_component_nodes).copy()

print(
    f"Giant component: "
    f"{graph_nx.number_of_nodes()} nodes, "
    f"{graph_nx.number_of_edges()} edges"
)

# ------------------------------------------------------------------
# Partition
# ------------------------------------------------------------------

if ALGORITHM == "kern":
    print("Running Kernighan-Lin...")
    side_0, side_1 = kernighan_lin_bisection(graph_nx)

elif ALGORITHM == "spectral":
    side_0, side_1 = spectral_partition(graph_nx)

elif ALGORITHM == "louvain":
    side_0, side_1 = louvain_partition(graph_nx)

elif ALGORITHM == "leiden":
    side_0, side_1 = leiden_partition(graph_nx)

else:
    raise ValueError(f"Unknown algorithm '{ALGORITHM}'")

# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

cut_edges = nx.cut_size(graph_nx, side_0, side_1)

n = graph_nx.number_of_nodes()

print("\nPartition statistics")
print("--------------------")
print(f"Side 0: {len(side_0)}")
print(f"Side 1: {len(side_1)}")
print(f"Ratio: {len(side_0) / n:.2%}")
print(f"Cut edges: {cut_edges}")

# ------------------------------------------------------------------
# Save graph
# ------------------------------------------------------------------

membership = build_membership(side_0, side_1)

nx.set_node_attributes(graph_nx, membership, "side")

nx.set_node_attributes(
    graph_nx, {node: ALGORITHM for node in graph_nx.nodes()}, "algorithm"
)

nx.write_graphml(graph_nx, str(PARTIT_GRAPH))

print(f"\nPartitioned GraphML saved to:\n{PARTIT_GRAPH}")
