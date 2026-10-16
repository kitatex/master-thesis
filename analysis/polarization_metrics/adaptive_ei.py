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


def compute_aei_from_counts(
    n_0: int,
    n_1: int,
    e_00: float,
    e_11: float,
    e_01: float,
    double_cross_density: bool = False,
) -> float:
    """
    Computes the Adaptive EI Index (AEI) (Chen et al., 2021) from partition node sizes
    (n_0, n_1) and undirected edge counts (e_00, e_11, e_01).

    D_00 = e_00 / (n_0 * (n_0 - 1) / 2)
    D_11 = e_11 / (n_1 * (n_1 - 1) / 2)
    D_01 = e_01 / (n_0 * n_1)

    If double_cross_density is False (standard undirected formulation):
        AEI = (D_00 + D_11 - D_01) / (D_00 + D_11 + D_01)
    If double_cross_density is True (literal 4-term D_12 + D_21 = 2 * D_01):
        AEI = (D_00 + D_11 - 2*D_01) / (D_00 + D_11 + 2*D_01)
    """
    if n_0 < 2 or n_1 < 2:
        return 0.0

    poss_00 = 0.5 * n_0 * (n_0 - 1)
    poss_11 = 0.5 * n_1 * (n_1 - 1)
    poss_01 = float(n_0 * n_1)

    d_00 = e_00 / poss_00
    d_11 = e_11 / poss_11
    d_01 = e_01 / poss_01

    cross_term = (2.0 * d_01) if double_cross_density else d_01
    denom = d_00 + d_11 + cross_term
    if denom <= 1e-12:
        return 0.0

    return float((d_00 + d_11 - cross_term) / denom)


def compute_aei_from_graph(G: nx.Graph, double_cross_density: bool = False) -> float:
    """Computes AEI directly from a partitioned NetworkX graph with node attribute 'side'."""
    G_undir = G.to_undirected()
    n_0 = sum(1 for _, d in G_undir.nodes(data=True) if int(d["side"]) == 0)
    n_1 = sum(1 for _, d in G_undir.nodes(data=True) if int(d["side"]) == 1)

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

    return compute_aei_from_counts(
        n_0, n_1, e_00, e_11, e_01, double_cross_density=double_cross_density
    )


if __name__ == "__main__":
    print(f"Loading partitioned graph: {INPUT_GRAPH.name}...")
    G = nx.read_graphml(INPUT_GRAPH)
    aei_std = compute_aei_from_graph(G, double_cross_density=False)
    aei_2x = compute_aei_from_graph(G, double_cross_density=True)
    print(f"Adaptive EI Index (Standard Undirected): {aei_std:.4f}")
    print(f"Adaptive EI Index (2x Cross-Density):    {aei_2x:.4f}")
