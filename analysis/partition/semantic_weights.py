import networkx as nx
import igraph as ig
from config.paths import TOPICS_PATH

TOPIC_NAME = "#gaza"
MIN_RP = 2
MIN_URL = 3
URL_SOURCE = "global"

# Focused on the algorithms that produced the best structural results
_ALLOWED_ALGORITHMS = ["eigenvector", "spinglass"]
ALGORITHM = "eigenvector"  # Change to 'eigenvector' to test the other

INPUT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/with_opinions/{TOPIC_NAME}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.graphml"
)

# #gaza
INPUT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_minrp{MIN_RP}_minurl{MIN_URL}_itp.graphml"
)

PARTIT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp{MIN_RP}_part_semantic_{ALGORITHM}.graphml"
)

PARTIT_GRAPH.parent.mkdir(parents=True, exist_ok=True)

# 1. Load and ensure undirected
print("Loading graph with opinion scores...")
graph_nx = nx.read_graphml(INPUT_GRAPH)
graph_nx = graph_nx.to_undirected()

# 2. Extract the Giant Component
components = list(nx.connected_components(graph_nx))
giant_component_nodes = max(components, key=len)
graph_nx = graph_nx.subgraph(giant_component_nodes).copy()
print(f"Extracted giant component with {graph_nx.number_of_nodes()} nodes.")

# 3. Calculate Semantic Edge Weights
print("Calculating semantic edge weights...")
for u, v in graph_nx.edges():
    score_u = graph_nx.nodes[u].get("url_bias_score")
    score_v = graph_nx.nodes[v].get("url_bias_score")

    if score_u is not None and score_v is not None:
        opinion_distance = abs(float(score_u) - float(score_v))
        # High agreement = high weight (closer to 1.0)
        # High disagreement = low weight (closer to 0.0)
        weight = 1.0 / (1.0 + opinion_distance)
    else:
        weight = 0.5  # Neutral fallback if data is missing

    graph_nx[u][v]["semantic_weight"] = weight

if ALGORITHM not in _ALLOWED_ALGORITHMS:
    print("Use a valid partitioning algorithm")

else:
    # 4. Convert to igraph (Preserving Weights and Attributes)
    print("Converting to igraph with semantic weights...")
    nodes = list(graph_nx.nodes())
    mapping = {node: i for i, node in enumerate(nodes)}
    inverse_mapping = {i: node for i, node in enumerate(nodes)}

    edges = [(mapping[u], mapping[v]) for u, v in graph_nx.edges()]

    # Extract the weights in the exact order of the edges list
    edge_weights = [graph_nx[u][v]["semantic_weight"] for u, v in graph_nx.edges()]

    graph_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
    graph_ig.vs["name"] = nodes
    graph_ig.es["weight"] = edge_weights

    # 5. Run Weighted Partitioning
    if ALGORITHM == "eigenvector":
        print("Running Semantic Leading Eigenvector...")
        partition = graph_ig.community_leading_eigenvector(clusters=2, weights="weight")

    elif ALGORITHM == "spinglass":
        print("Running Semantic Spinglass...")
        # Spinglass has a resolution parameter 'gamma' (default 1.0).
        # Lowering it slightly (e.g., 0.9) can sometimes help capture smaller minority communities.
        partition = graph_ig.community_spinglass(spins=2, weights="weight")

    # 6. Map back to NetworkX and Save
    membership = {}
    for i, comm in enumerate(partition):
        for vertex_index in comm:
            original_node_name = inverse_mapping[vertex_index]
            membership[original_node_name] = i

    nx.set_node_attributes(graph_nx, membership, "side")
    nx.write_graphml(graph_nx, str(PARTIT_GRAPH))
    print(f"Partitioned GraphML saved to: {PARTIT_GRAPH}")
