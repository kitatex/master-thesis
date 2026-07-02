import random
import statistics
import networkx as nx

from config.paths import TOPICS_PATH, RWC_PATH

PARTIT_GRAPH = ""


REPRODUCE_PAPER = False

if REPRODUCE_PAPER:
    # controversial:
    # retweet_graph_beefban_threshold_largest_CC_undir_gcc.graphml
    # retweet_graph_russia_march_threshold_largest_CC_undir_gcc.graphml
    # uncontroversial:
    # retweet_graph_sxsw_threshold_largest_CC_undir_gcc.graphml
    # retweet_graph_germanwings_threshold_largest_CC_undir_gcc.graphml
    FILE = "retweet_graph_germanwings_threshold_largest_CC_undir_gcc.graphml"
    PARTIT_GRAPH = RWC_PATH / f"graphs_graphml_selection/partit/{FILE}"
else:
    TOPIC_NAME = "#ukraine"

    FILE = (
        f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp2_part_leidenmeta0.3.graphml"
    )

    PARTIT_GRAPH = TOPICS_PATH / FILE


G = nx.read_graphml(PARTIT_GRAPH)

left_nodes = {node for node, data in G.nodes(data=True) if int(data["side"]) == 0}
right_nodes = {node for node, data in G.nodes(data=True) if int(data["side"]) == 1}


def random_walk_until_seed(start_node, left_seeds, right_seeds):
    current = start_node

    while True:
        current = random.choice(list(G.neighbors(current)))

        if current in left_seeds:
            return "left"

        if current in right_seeds:
            return "right"


def sample_seed_nodes(nodes, k):
    return set(random.sample(list(nodes), k))


def rwc_single_run(
    left_nodes,
    right_nodes,
    seed_fraction=0.10,
):
    left_seed_count = max(1, int(seed_fraction * len(left_nodes)))
    right_seed_count = max(1, int(seed_fraction * len(right_nodes)))

    left_seeds = sample_seed_nodes(left_nodes, left_seed_count)
    right_seeds = sample_seed_nodes(right_nodes, right_seed_count)

    left_left = 0
    left_right = 0
    right_left = 0
    right_right = 0

    #
    # Start from left seeds
    #
    for start in left_seeds:
        other_left = left_seeds - {start}

        side = random_walk_until_seed(
            start,
            other_left,
            right_seeds,
        )

        if side == "left":
            left_left += 1
        else:
            left_right += 1

    #
    # Start from right seeds
    #
    for start in right_seeds:
        other_right = right_seeds - {start}

        side = random_walk_until_seed(
            start,
            left_seeds,
            other_right,
        )

        if side == "right":
            right_right += 1
        else:
            right_left += 1

    #
    # Same equations as authors' code
    #
    e1 = left_left / (left_left + right_left)
    e2 = left_right / (left_right + right_right)
    e3 = right_left / (left_left + right_left)
    e4 = right_right / (left_right + right_right)

    rwc = e1 * e4 - e2 * e3

    return rwc


def rwc_statistics(
    left_nodes,
    right_nodes,
    seed_fraction=0.10,
    repetitions=100,
):
    scores = []

    for _ in range(repetitions):
        score = rwc_single_run(
            set(left_nodes),
            set(right_nodes),
            seed_fraction=seed_fraction,
        )
        scores.append(score)

    return {
        "mean": statistics.mean(scores),
        "std": statistics.stdev(scores),
        "min": min(scores),
        "max": max(scores),
        "median": statistics.median(scores),
        "all_scores": scores,
    }


stats = rwc_statistics(
    left_nodes,
    right_nodes,
    seed_fraction=0.10,
    repetitions=500,
)

print("MC-based rwc for: \t", FILE)
print(f"Mean   : {stats['mean']:.4f}")
print(f"Std    : {stats['std']:.4f}")
print(f"Median : {stats['median']:.4f}")
print(f"Min    : {stats['min']:.4f}")
print(f"Max    : {stats['max']:.4f}")
