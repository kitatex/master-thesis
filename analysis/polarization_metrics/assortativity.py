import networkx as nx
from config.paths import TOPICS_PATH

# --- Standalone Configuration ---
TOPIC_NAME = "immigration"
MIN_RP = 2
ALGORITHM = "leidenmeta"
LEIDEN_RESOLUTION = 1

INPUT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp{MIN_RP}_part_{ALGORITHM}{LEIDEN_RESOLUTION}.graphml"
)


def compute_assortativity_from_counts(e_00: float, e_11: float, e_01: float) -> float:
    """
    Computes Newman's (2003) Assortativity Coefficient (AC) for an undirected
    2-partitioned graph from internal edge weights/counts (e_00, e_11) and
    cross-partition edge weight/count (e_01).

    Formula:
        AC = (Tr(e) - ||e^2||) / (1 - ||e^2||)
    """
    total_edges = e_00 + e_11 + e_01
    if total_edges <= 0:
        return 0.0

    # Mixing matrix elements (fraction of directed stubs)
    e11 = e_00 / total_edges
    e22 = e_11 / total_edges
    e12 = (0.5 * e_01) / total_edges
    e21 = e12

    tr_e = e11 + e22
    a1 = e11 + e12
    a2 = e22 + e21
    sum_a_sq = (a1 * a1) + (a2 * a2)

    denom = 1.0 - sum_a_sq
    if denom <= 1e-12:
        return 0.0

    return float((tr_e - sum_a_sq) / denom)


def compute_assortativity_from_graph(G: nx.Graph) -> float:
    """Computes AC directly from a partitioned NetworkX graph with node attribute 'side'."""
    G_undir = G.to_undirected()
    e_00 = 0.0
    e_11 = 0.0
    e_01 = 0.0

    for u, v, data in G_undir.edges(data=True):
        s_u = int(G_undir.nodes[u]["side"])
        s_v = int(G_undir.nodes[v]["side"])
        w = float(data.get("weight", 1.0))
        if s_u == 0 and s_v == 0:
            e_00 += w
        elif s_u == 1 and s_v == 1:
            e_11 += w
        else:
            e_01 += w

    return compute_assortativity_from_counts(e_00, e_11, e_01)


if __name__ == "__main__":
    print(f"Loading partitioned graph: {INPUT_GRAPH.name}...")
    G = nx.read_graphml(INPUT_GRAPH)
    ac_score = compute_assortativity_from_graph(G)
    print(f"Assortativity Coefficient (AC): {ac_score:.4f}")
