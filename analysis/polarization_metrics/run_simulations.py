import warnings
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
from tqdm import tqdm

from config.paths import TOPICS_PATH

from analysis.polarization_metrics.welch_t_test import compute_welch_arrays
from analysis.partition.leiden_meta import (
    get_opinion_graph_path,
    extract_giant_component,
    build_community_metagraph,
    spectral_partition_graph,
)
from analysis.polarization_metrics.rwc_vectorized import (
    precompute_adjacency,
    compute_rwc_from_adj,
)
from analysis.polarization_metrics.point_biserial import (
    is_node_interpolated,
    compute_point_biserial_arrays,
)
from analysis.polarization_metrics.assortativity import (
    compute_assortativity_from_counts,
)
from analysis.polarization_metrics.adaptive_ei import (
    compute_aei_from_counts,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# CONFIGURATION
# ==========================================
TOPIC_GROUPS = {
    "Political": [
        "capitalism",
        "climate change",
        "covid-19",
        "immigration",
        "gaza",
        "ukraine",
    ],
    "Non-Controversial (Control)": [
        "0 basketball",
        "0 bg3",
        "0 coffee",
        "0 game development",
        "0 photography",
        "0 pokemon",
    ],
}

MIN_RP = 2
MIN_URL = 3
URL_SOURCE = "global"
INTERPOLATED = True

# Default Leiden modularity resolution (gamma = 1.0)
LEIDEN_RESOLUTION = 1.0
N_SIMULATIONS = 1000

# RWC settings per simulation run:
# Since we already run 100 outer simulations per topic, 10 RWC seed resamples x 1000 walks
# per partition = 1,000,000 random walks per topic across the 100 runs!
RWC_REPETITIONS_PER_PARTITION = 10
RWC_WALKS_PER_SIDE = 1000
RWC_SEED_FRACTION = 0.10

OUTPUT_DIR = (TOPICS_PATH / "_polarization_results").resolve()
OUTPUT_CSV = OUTPUT_DIR / f"polarization_simulations_n{N_SIMULATIONS}.csv"


# ==========================================
# PREPARATION & SINGLE SIMULATION STEP
# ==========================================
def prepare_topic_graph(graph_path):
    """
    Loads the topic graph once, extracts the giant component via leiden_meta,
    and pre-computes integer node indices (0..N-1) for fast repeated partitioning.
    """
    raw_G = nx.read_graphml(graph_path)
    G = extract_giant_component(raw_G)

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    G_int = nx.relabel_nodes(G, node_to_idx, copy=True)

    edges_weighted = [
        (u, v, float(G_int[u][v].get("weight", 1.0))) for u, v in G_int.edges()
    ]
    g_ig = ig.Graph(
        n=len(nodes),
        edges=[(u, v) for u, v, _ in edges_weighted],
        directed=False,
    )

    adj_dict = precompute_adjacency(G_int)

    all_valid_idx, all_opinions = [], []
    seed_valid_idx, seed_opinions = [], []

    for i in range(len(nodes)):
        node_attr = G_int.nodes[i]
        score = node_attr.get("url_bias_score")
        if score is not None:
            val = float(score)
            all_valid_idx.append(i)
            all_opinions.append(val)
            if not is_node_interpolated(node_attr):
                seed_valid_idx.append(i)
                seed_opinions.append(val)

    return {
        "g_ig": g_ig,
        "edges_weighted": edges_weighted,
        "adj_dict": adj_dict,
        "all_valid_idx": np.array(all_valid_idx, dtype=int),
        "all_opinions": np.array(all_opinions, dtype=float),
        "seed_valid_idx": np.array(seed_valid_idx, dtype=int),
        "seed_opinions": np.array(seed_opinions, dtype=float),
    }


def simulate_single_partition_and_measure(
    prepped: dict, resolution: float, seed: int
) -> dict | None:
    """
    Executes 1 stochastic Leiden-Metagraph partition and evaluates:
      1. Point-Biserial Correlation (r_pb & r_pb_seed_only)
      2. Random Walk Controversy (rwc)
      3. Assortativity Coefficient (ac)
      4. Adaptive EI Index (aei & aei_2x_cross)
    """
    g_ig = prepped["g_ig"]
    n_total = g_ig.vcount()

    # 1. Partition via Leiden + Spectral Metagraph Bisection
    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    n_comms = len(partition)
    if n_comms < 2:
        return None

    comm_graph = build_community_metagraph(
        n_comms, partition.membership, prepped["edges_weighted"]
    )
    try:
        meta_side_0, _ = spectral_partition_graph(comm_graph)
    except ValueError:
        return None

    membership = partition.membership
    node_sides = np.array(
        [0 if membership[i] in meta_side_0 else 1 for i in range(n_total)],
        dtype=int,
    )

    # Canonically orient so Side 0 always has the lower mean opinion score
    all_idx = prepped["all_valid_idx"]
    all_ops = prepped["all_opinions"]
    if len(all_idx) > 0:
        s0_mask = node_sides[all_idx] == 0
        s1_mask = node_sides[all_idx] == 1
        if np.any(s0_mask) and np.any(s1_mask):
            if np.mean(all_ops[s0_mask]) > np.mean(all_ops[s1_mask]):
                node_sides = 1 - node_sides

    side_0_idx = np.where(node_sides == 0)[0].tolist()
    side_1_idx = np.where(node_sides == 1)[0].tolist()
    n_0 = len(side_0_idx)
    n_1 = len(side_1_idx)

    if n_0 == 0 or n_1 == 0:
        return None

    # 2. Count internal (e_00, e_11) and crossing (e_01) edges
    e_00 = 0.0
    e_11 = 0.0
    e_01 = 0.0
    for u, v, w in prepped["edges_weighted"]:
        su = node_sides[u]
        sv = node_sides[v]
        if su == 0 and sv == 0:
            e_00 += w
        elif su == 1 and sv == 1:
            e_11 += w
        else:
            e_01 += w

    # 3. Compute Structural Polarization Measures (AC, AEI, RWC)
    ac_score = compute_assortativity_from_counts(e_00, e_11, e_01)
    aei_score = compute_aei_from_counts(
        n_0, n_1, e_00, e_11, e_01, double_cross_density=False
    )
    aei_2x_score = compute_aei_from_counts(
        n_0, n_1, e_00, e_11, e_01, double_cross_density=True
    )

    rwc_stats = compute_rwc_from_adj(
        prepped["adj_dict"],
        side_0_idx,
        side_1_idx,
        seed_fraction=RWC_SEED_FRACTION,
        repetitions=RWC_REPETITIONS_PER_PARTITION,
        walks_per_side=RWC_WALKS_PER_SIDE,
        base_seed=seed,
    )
    rwc_score = rwc_stats["mean"]
    rwc_internal_std = rwc_stats["std"]

    # 4. Compute Opinion-Structural Polarization Measure (Point-Biserial & Welch's t-test)
    r_pb_all, p_val_all = compute_point_biserial_arrays(
        node_sides[all_idx], all_ops, canonical_orient=True
    )
    r_pb_seed, p_val_seed = compute_point_biserial_arrays(
        node_sides[prepped["seed_valid_idx"]],
        prepped["seed_opinions"],
        canonical_orient=True,
    )

    welch_t_all, welch_p_all, opinion_diff_all = compute_welch_arrays(
        node_sides[all_idx], all_ops, canonical_orient=True
    )
    welch_t_seed, welch_p_seed, opinion_diff_seed = compute_welch_arrays(
        node_sides[prepped["seed_valid_idx"]],
        prepped["seed_opinions"],
        canonical_orient=True,
    )

    return {
        "r_pb": r_pb_all,
        "p_value": p_val_all,
        "r_pb_seed_only": r_pb_seed,
        "p_value_seed_only": p_val_seed,
        "welch_t": welch_t_all,
        "welch_p": welch_p_all,
        "opinion_diff": opinion_diff_all,
        "welch_t_seed_only": welch_t_seed,
        "welch_p_seed_only": welch_p_seed,
        "opinion_diff_seed_only": opinion_diff_seed,
        "rwc": rwc_score,
        "rwc_mc_std": rwc_internal_std,
        "ac": ac_score,
        "aei": aei_score,
        "aei_2x_cross": aei_2x_score,
        "n_communities": n_comms,
        "n_side_0": n_0,
        "n_side_1": n_1,
        "minority_ratio": min(n_0, n_1) / n_total,
        "e_00": e_00,
        "e_11": e_11,
        "e_01_cut": e_01,
    }


def run_all_simulations():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for group_name, topic_list in TOPIC_GROUPS.items():
        for topic in topic_list:
            graph_path = get_opinion_graph_path(
                topic, URL_SOURCE, MIN_RP, MIN_URL, INTERPOLATED
            )
            if not graph_path.exists():
                print(f"[WARNING] Skipping '{topic}': {graph_path} not found")
                continue

            print(f"\nPreparing '{topic}' [{group_name}]...")
            prepped = prepare_topic_graph(graph_path)

            for sim_idx in tqdm(range(N_SIMULATIONS), desc=f"Simulating {topic}"):
                seed = 1000 + sim_idx
                metrics = simulate_single_partition_and_measure(
                    prepped, resolution=LEIDEN_RESOLUTION, seed=seed
                )
                if metrics is not None:
                    records.append(
                        {
                            "group": group_name,
                            "topic": topic,
                            "simulation": sim_idx,
                            "seed": seed,
                            "resolution": LEIDEN_RESOLUTION,
                            **metrics,
                        }
                    )

    df = pd.DataFrame(records)
    with open(OUTPUT_CSV.resolve(), "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)

    print("\n" + "=" * 65)
    print(
        f"Completed {len(df)} total simulation runs across {df['topic'].nunique()} topics."
    )
    print(f"Saved results CSV to:\n{OUTPUT_CSV.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    run_all_simulations()
