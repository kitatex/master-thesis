import warnings
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
from tqdm import tqdm

from config.paths import TOPICS_PATH

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
        "war in gaza",
        "war in ukraine",
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

RESOLUTIONS = np.round(np.arange(0.1, 2.05, 0.1), 2)
N_SIMULATIONS = 25  # Leiden runs per (topic, resolution)

RWC_REPETITIONS_PER_RUN = 5
RWC_WALKS_PER_SIDE = 500
RWC_SEED_FRACTION = 0.10

OUTPUT_DIR = (TOPICS_PATH / "_hyperparameter_tuning").resolve()
OUTPUT_CSV = OUTPUT_DIR / "unsupervised_leiden_calibration.csv"


# ==========================================
# GRAPH PREPARATION & SIMULATION
# ==========================================
def prepare_graph_data(graph_path):
    """
    Loads the graph via leiden_meta.extract_giant_component and pre-indexes
    nodes as 0..N-1 for fast in-memory Leiden + RWC + Point-Biserial evaluation.
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
    degrees = np.array(g_ig.degree(), dtype=float)
    total_volume = float(np.sum(degrees))

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
        "degrees": degrees,
        "total_volume": total_volume,
        "all_valid_idx": np.array(all_valid_idx, dtype=int),
        "all_opinions": np.array(all_opinions, dtype=float),
        "seed_valid_idx": np.array(seed_valid_idx, dtype=int),
        "seed_opinions": np.array(seed_opinions, dtype=float),
    }


def evaluate_single_run(prepped: dict, resolution: float, seed: int):
    g_ig = prepped["g_ig"]

    # 1. Leiden Community Detection
    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    n_comms = len(partition)
    if n_comms < 2:
        return None

    # 2. Build Metagraph & Bisect
    comm_graph = build_community_metagraph(
        n_comms, partition.membership, prepped["edges_weighted"]
    )
    try:
        meta_side_0, _ = spectral_partition_graph(comm_graph)
    except ValueError:
        return None

    membership = partition.membership
    side_0_idx = [i for i in range(g_ig.vcount()) if membership[i] in meta_side_0]
    side_1_idx = [i for i in range(g_ig.vcount()) if membership[i] not in meta_side_0]

    if not side_0_idx or not side_1_idx:
        return None

    node_sides = np.zeros(g_ig.vcount(), dtype=int)
    node_sides[side_1_idx] = 1

    # 3. Unsupervised Topological Metrics
    n_total = g_ig.vcount()
    p0 = len(side_0_idx) / n_total
    p1 = len(side_1_idx) / n_total
    minority_ratio = min(p0, p1)
    balance_factor = 4.0 * p0 * p1

    cut_size = sum(
        data["weight"]
        for u, v, data in comm_graph.edges(data=True)
        if (u in meta_side_0) != (v in meta_side_0)
    )
    vol_0 = float(np.sum(prepped["degrees"][side_0_idx]))
    vol_1 = prepped["total_volume"] - vol_0
    ncut = (cut_size / vol_0 + cut_size / vol_1) if (vol_0 > 0 and vol_1 > 0) else 2.0

    rwc_stats = compute_rwc_from_adj(
        prepped["adj_dict"],
        side_0_idx,
        side_1_idx,
        seed_fraction=RWC_SEED_FRACTION,
        repetitions=RWC_REPETITIONS_PER_RUN,
        walks_per_side=RWC_WALKS_PER_SIDE,
        base_seed=seed,
    )
    rwc_score = rwc_stats["mean"]
    topo_quality = rwc_score * balance_factor

    # 4. Downstream Ideological Evaluation
    r_pb_all, p_val_all = compute_point_biserial_arrays(
        node_sides[prepped["all_valid_idx"]],
        prepped["all_opinions"],
        canonical_orient=True,
    )
    r_pb_seed, p_val_seed = compute_point_biserial_arrays(
        node_sides[prepped["seed_valid_idx"]],
        prepped["seed_opinions"],
        canonical_orient=True,
    )

    return {
        "rwc": rwc_score,
        "minority_ratio": minority_ratio,
        "balance_factor": balance_factor,
        "ncut": ncut,
        "topo_quality": topo_quality,
        "r_pb": r_pb_all,
        "p_value": p_val_all,
        "r_pb_seed_only": r_pb_seed,
        "p_value_seed_only": p_val_seed,
        "n_communities": n_comms,
    }


def run_unsupervised_calibration():
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

            print(f"\nLoading '{topic}' [{group_name}] from {graph_path.name}...")
            prepped = prepare_graph_data(graph_path)

            total_iters = len(RESOLUTIONS) * N_SIMULATIONS
            with tqdm(total=total_iters, desc=f"Simulating {topic}") as pbar:
                for res in RESOLUTIONS:
                    for sim_idx in range(N_SIMULATIONS):
                        seed = 42 + sim_idx * 1000 + int(res * 100)
                        metrics = evaluate_single_run(
                            prepped, resolution=float(res), seed=seed
                        )
                        if metrics is not None:
                            records.append(
                                {
                                    "group": group_name,
                                    "topic": topic,
                                    "resolution": float(res),
                                    "simulation": sim_idx,
                                    **metrics,
                                }
                            )
                        pbar.update(1)

    df = pd.DataFrame(records)
    with open(OUTPUT_CSV.resolve(), "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
    print(f"\nSaved calibration simulation data to:\n{OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    run_unsupervised_calibration()
