import networkx as nx
from config.paths import TOPICS_PATH

TOPIC_NAME = "immigration"
MIN_RP = 2
MIN_URL = 3
URL_SOURCE = "global"

THRESHOLD = -0.53

INPUT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/with_opinions/{TOPIC_NAME}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.graphml"
)

PARTIT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp{MIN_RP}_part_threshold.graphml"
)

PARTIT_GRAPH.parent.mkdir(parents=True, exist_ok=True)

# 1. Load and ensure undirected
print("Loading graph with opinion scores...")
graph_nx = nx.read_graphml(INPUT_GRAPH)
graph_nx = graph_nx.to_undirected()

# 2. Extract the Giant Component
components = list(nx.connected_components(graph_nx))
if components:
    giant_component_nodes = max(components, key=len)
    graph_nx = graph_nx.subgraph(giant_component_nodes).copy()
    print(f"Extracted giant component with {graph_nx.number_of_nodes()} nodes.")

# 3. Apply the Threshold Cut
print(f"Applying semantic threshold cut at {THRESHOLD}...")
membership = {}
missing_scores = 0

for node, data in graph_nx.nodes(data=True):
    score = data.get("url_bias_score")

    if score is not None:
        # The core threshold logic
        if float(score) > THRESHOLD:
            membership[node] = 1
        else:
            membership[node] = 0
    else:
        # Fallback for nodes missing a score
        membership[node] = 0
        missing_scores += 1

if missing_scores > 0:
    print(
        f"WARNING: {missing_scores} nodes had no 'url_bias_score' and were defaulted to side 0."
    )

# 4. Save the Partitioned Graph
nx.set_node_attributes(graph_nx, membership, "side")
nx.write_graphml(graph_nx, str(PARTIT_GRAPH))
print(f"Partitioned GraphML saved to: {PARTIT_GRAPH}")
