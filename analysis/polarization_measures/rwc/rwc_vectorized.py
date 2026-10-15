import random
import statistics
import networkx as nx
from config.paths import TOPICS_PATH, RWC_PATH

REPRODUCE_PAPER = False

if REPRODUCE_PAPER:
    FILE = "retweet_graph_germanwings_threshold_largest_CC_undir_gcc.graphml"
    PARTIT_GRAPH = RWC_PATH / f"graphs_graphml_selection/partit/{FILE}"
else:
    TOPIC_NAME = "0 pokemon"
    FILE = (
        f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp2_part_leidenmeta0.3.graphml"
    )
    PARTIT_GRAPH = TOPICS_PATH / FILE


def precompute_adjacency(G: nx.Graph) -> dict:
    """
    Pre-converts NetworkX neighbors into tuples for O(1) random.choice() speed.
    """
    return {node: tuple(G.neighbors(node)) for node in G.nodes()}


def random_walk_until_seed(
    start_node,
    adj_dict: dict,
    left_seeds: set,
    right_seeds: set,
    rng: random.Random,
    max_steps: int = 10000,
):
    current = start_node
    for _ in range(max_steps):
        neighbors = adj_dict[current]
        if not neighbors:
            return None
        current = rng.choice(neighbors)
        if current in left_seeds:
            return "left"
        if current in right_seeds:
            return "right"
    return None


def rwc_single_run(
    adj_dict: dict,
    left_list: list,
    right_list: list,
    seed_fraction: float = 0.10,
    walks_per_side: int = 1000,
    rng: random.Random | None = None,
) -> float:
    """
    Performs one Monte Carlo iteration of Garimella et al.'s Random Walk Controversy.
    """
    if not left_list or not right_list:
        return 0.0

    if rng is None:
        rng = random.Random()

    left_seed_count = max(1, int(seed_fraction * len(left_list)))
    right_seed_count = max(1, int(seed_fraction * len(right_list)))

    left_seeds = set(rng.sample(left_list, left_seed_count))
    right_seeds = set(rng.sample(right_list, right_seed_count))

    left_left = 0
    left_right = 0
    right_left = 0
    right_right = 0

    # 1. Start `walks_per_side` walks uniformly from Side X (Left)
    for _ in range(walks_per_side):
        start = rng.choice(left_list)
        active_left_seeds = left_seeds - {start} if len(left_seeds) > 1 else left_seeds
        side = random_walk_until_seed(
            start, adj_dict, active_left_seeds, right_seeds, rng
        )
        if side == "left":
            left_left += 1
        elif side == "right":
            left_right += 1

    # 2. Start `walks_per_side` walks uniformly from Side Y (Right)
    for _ in range(walks_per_side):
        start = rng.choice(right_list)
        active_right_seeds = (
            right_seeds - {start} if len(right_seeds) > 1 else right_seeds
        )
        side = random_walk_until_seed(
            start, adj_dict, left_seeds, active_right_seeds, rng
        )
        if side == "right":
            right_right += 1
        elif side == "left":
            right_left += 1

    total_left_walks = left_left + left_right
    total_right_walks = right_left + right_right
    if total_left_walks == 0 or total_right_walks == 0:
        return 0.0

    p_end_L_given_start_L = left_left / total_left_walks
    p_end_R_given_start_L = left_right / total_left_walks
    p_end_L_given_start_R = right_left / total_right_walks
    p_end_R_given_start_R = right_right / total_right_walks

    denom_L = p_end_L_given_start_L + p_end_L_given_start_R
    denom_R = p_end_R_given_start_L + p_end_R_given_start_R

    e1 = p_end_L_given_start_L / denom_L if denom_L > 0 else 0.5
    e3 = p_end_L_given_start_R / denom_L if denom_L > 0 else 0.5
    e2 = p_end_R_given_start_L / denom_R if denom_R > 0 else 0.5
    e4 = p_end_R_given_start_R / denom_R if denom_R > 0 else 0.5

    return float((e1 * e4) - (e2 * e3))


def compute_rwc_from_adj(
    adj_dict: dict,
    left_list: list,
    right_list: list,
    seed_fraction: float = 0.10,
    repetitions: int = 100,
    walks_per_side: int = 1000,
    base_seed: int = 42,
) -> dict:
    """
    Reusable in-memory RWC evaluator that works on any precomputed adjacency dict.
    """
    scores = []
    for i in range(repetitions):
        rng = random.Random(base_seed + i)
        score = rwc_single_run(
            adj_dict, left_list, right_list, seed_fraction, walks_per_side, rng=rng
        )
        scores.append(score)

    return {
        "mean": statistics.mean(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
        "median": statistics.median(scores),
        "all_scores": scores,
    }


def rwc_statistics(
    G: nx.Graph,
    seed_fraction: float = 0.10,
    repetitions: int = 100,
    walks_per_side: int = 1000,
    base_seed: int = 42,
) -> dict:
    G_undir = G.to_undirected()
    adj_dict = precompute_adjacency(G_undir)

    left_list = [
        node for node, data in G_undir.nodes(data=True) if int(data["side"]) == 0
    ]
    right_list = [
        node for node, data in G_undir.nodes(data=True) if int(data["side"]) == 1
    ]

    return compute_rwc_from_adj(
        adj_dict,
        left_list,
        right_list,
        seed_fraction,
        repetitions,
        walks_per_side,
        base_seed,
    )


if __name__ == "__main__":
    G = nx.read_graphml(PARTIT_GRAPH)
    stats = rwc_statistics(G, seed_fraction=0.10, repetitions=100, walks_per_side=1000)

    print("MC-based RWC for:\t", FILE)
    print(f"Mean   : {stats['mean']:.4f}")
    print(f"Std    : {stats['std']:.4f}")
    print(f"Median : {stats['median']:.4f}")
    print(f"Min    : {stats['min']:.4f}")
    print(f"Max    : {stats['max']:.4f}")
