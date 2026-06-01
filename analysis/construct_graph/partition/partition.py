import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

from config.paths import TOPICS_PATH

TOPIC_NAME = "#gaza"
MIN_RP = 1
MIN_URL = 2
URL_SOURCE = "global"  # "global" or "discussion"

INPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.graphml"
)


OUTPUT_DIR = TOPICS_PATH / f"{TOPIC_NAME}/graph/partition"

OUTPUT_GRAPH_PARTIT = OUTPUT_DIR / f"{TOPIC_NAME}_partit_minrp{MIN_RP}.graphml"

graph = nx.read_graphml(INPUT_GRAPHML)

# Convert igraph -> NetworkX
graph_nx = nx.Graph()
graph_nx.add_nodes_from(range(graph.vcount()))
graph_nx.add_edges_from(graph.get_edgelist())

print("Kernighan-Lin Bisection...")
side_0, side_1 = kernighan_lin_bisection(graph_nx)

# Convert partition into membership vector
membership = [0] * graph.vcount()
for node in side_1:
    membership[node] = 1
# Assign the resulting sides (0 or 1) as a vertex attribute
graph.vs["side"] = membership
graph.write_graphml(str(OUTPUT_GRAPH_PARTIT))
print(f"Partitioned GraphML saved to: {OUTPUT_GRAPH_PARTIT}")
