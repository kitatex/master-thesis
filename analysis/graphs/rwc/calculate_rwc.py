import igraph as ig

from config.paths import TOPICS_PATH, RWC_PATH

TOPIC_NAME = "#aiethics"  # e.g., #climatecrisis

PATH_PARTITIONED_GRAPH = (
    TOPICS_PATH / f"{TOPIC_NAME}/graph/network_partitioned_k2.graphml"
)

PATH_PARTITIONED_GRAPH = (
    RWC_PATH
    / "graphs_graphml/retweet_graph_sxsw_threshold_largest_CC_partitioned_k2.graphml"
)


def calculate_rwc_rwr(
    g: ig.Graph, side_X: list, side_Y: list, sample_percent: float = 0.05
):
    """
    Calculates the Random Walk Controversy (RWC) score using the Efficient
    Random Walk with Restart (RWR) variant as defined by Garimella et al. (2018).
    """

    # Create a copy of the graph because we must modify its structure for the RWR
    g_mod = g.copy()
    V_count = g_mod.vcount()

    # 1. IDENTIFY AUTHORITATIVE TARGETS (Highest In-Degree)
    # Using in-degree as a proxy for endorsements/authoritativeness
    degrees_X = g_mod.degree(side_X, mode="in")
    degrees_Y = g_mod.degree(side_Y, mode="in")

    k_X = max(1, int(len(side_X) * sample_percent))
    k_Y = max(1, int(len(side_Y) * sample_percent))

    # Extract the top k nodes (X+ and Y+ in the paper's notation)
    top_X = [
        v
        for v, d in sorted(zip(side_X, degrees_X), key=lambda x: x[1], reverse=True)[
            :k_X
        ]
    ]
    top_Y = [
        v
        for v, d in sorted(zip(side_Y, degrees_Y), key=lambda x: x[1], reverse=True)[
            :k_Y
        ]
    ]
    top_all = set(top_X + top_Y)

    # 2. MODIFY THE GRAPH: FORCE RESTARTS
    # Transform high-degree vertices into dangling vertices by removing outgoing edges
    edges_to_delete = []
    for v in top_all:
        edges_to_delete.extend(g_mod.incident(v, mode="out"))
    g_mod.delete_edges(edges_to_delete)

    # 3. SET RESTART DISTRIBUTIONS (Walk Origins)
    # P1 starts uniformly over X; P2 starts uniformly over Y
    reset_X = [0.0] * V_count
    reset_Y = [0.0] * V_count

    for v in side_X:
        reset_X[v] = 1.0 / len(side_X)
    for v in side_Y:
        reset_Y[v] = 1.0 / len(side_Y)

    # 4. COMPUTE STATIONARY DISTRIBUTIONS (Personalized PageRank)
    P1 = g_mod.personalized_pagerank(directed=True, reset=reset_X)
    P2 = g_mod.personalized_pagerank(directed=True, reset=reset_Y)

    # 5. SUM PROBABILITIES OVER TARGET SETS
    S1_X_plus = sum(P1[v] for v in top_X)
    S2_X_plus = sum(P2[v] for v in top_X)

    S1_Y_plus = sum(P1[v] for v in top_Y)
    S2_Y_plus = sum(P2[v] for v in top_Y)

    # 6. CALCULATE CONDITIONAL PROBABILITIES (Bayes' logic fixed)
    W_X = len(side_X) / V_count
    W_Y = len(side_Y) / V_count

    denom_X_plus = (W_X * S1_X_plus) + (W_Y * S2_X_plus)
    denom_Y_plus = (W_X * S1_Y_plus) + (W_Y * S2_Y_plus)

    # P_AB = Pr[start = A | end = B+]
    P_XX_plus = (W_X * S1_X_plus) / denom_X_plus if denom_X_plus > 0 else 0
    P_YX_plus = (W_Y * S2_X_plus) / denom_X_plus if denom_X_plus > 0 else 0

    P_YY_plus = (W_Y * S2_Y_plus) / denom_Y_plus if denom_Y_plus > 0 else 0
    P_XY_plus = (W_X * S1_Y_plus) / denom_Y_plus if denom_Y_plus > 0 else 0

    # 7. FINAL RWC SCORE
    rwc_score = (P_XX_plus * P_YY_plus) - (P_XY_plus * P_YX_plus)

    return rwc_score, P_XX_plus, P_YY_plus


if __name__ == "__main__":
    print(f"Loading graph from {PATH_PARTITIONED_GRAPH}...")
    g = ig.Graph.Read_GraphML(str(PATH_PARTITIONED_GRAPH))

    side_X = [v.index for v in g.vs if int(v["side"]) == 0]
    side_Y = [v.index for v in g.vs if int(v["side"]) == 1]

    print(f"Side X size: {len(side_X)}")
    print(f"Side Y size: {len(side_Y)}")

    print(f"Calculating RWR Controversy Score for {PATH_PARTITIONED_GRAPH}...")
    rwc, pXX, pYY = calculate_rwc_rwr(g, side_X, side_Y, sample_percent=0.05)

    print(f"P(Start X | End X+): {pXX:.4f}")
    print(f"P(Start Y | End Y+): {pYY:.4f}")
    print(f"Final Controversy Score (RWC): {rwc:.4f}")
