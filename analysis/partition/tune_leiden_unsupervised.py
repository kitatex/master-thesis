import warnings
import random
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
from scipy.sparse.linalg import eigsh
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config.paths import TOPICS_PATH

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
GRAPH_SUFFIX = "_interpolated.graphml"

RESOLUTIONS = np.round(np.arange(0.1, 2.05, 0.1), 2)
N_SIMULATIONS = 25  # Leiden runs per (topic, resolution)

OUTPUT_DIR = TOPICS_PATH / "_hyperparameter_tuning"
OUTPUT_CSV = OUTPUT_DIR / "unsupervised_leiden_calibration.csv"
OUTPUT_PLOT = OUTPUT_DIR / "unsupervised_leiden_calibration_plot.png"


# ==========================================
# TOPOLOGICAL & RWC FUNCTIONS
# ==========================================
def get_graph_path(topic: str) -> Path:
    base_dir = TOPICS_PATH / f"{topic}/graph/with_opinions"
    primary = (
        base_dir / f"{topic}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}{GRAPH_SUFFIX}"
    )
    if primary.exists():
        return primary
    return base_dir / f"{topic}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.graphml"


def calculate_rwc_rwr_fast(
    g_undir: ig.Graph,
    degrees: np.ndarray,
    side_0: list,
    side_1: list,
    sample_percent: float = 0.05,
) -> float:
    """
    Exact deterministic RWC (Garimella et al. 2018) with the undirected graph fix:
    Converts to mutual directed graph before removing outgoing edges from high-degree seeds.
    """
    if len(side_0) < 2 or len(side_1) < 2:
        return 0.0

    g_dir = g_undir.copy()
    g_dir.to_directed(mode="mutual")
    V_count = g_dir.vcount()

    k_0 = max(1, int(len(side_0) * sample_percent))
    k_1 = max(1, int(len(side_1) * sample_percent))

    # Sort by degree to pick top k authoritative nodes on each side
    top_0 = sorted(side_0, key=lambda v: degrees[v], reverse=True)[:k_0]
    top_1 = sorted(side_1, key=lambda v: degrees[v], reverse=True)[:k_1]
    top_all = set(top_0 + top_1)

    # Delete ONLY outgoing edges from target nodes so walks absorb/restart there
    edges_to_delete = []
    for v in top_all:
        edges_to_delete.extend(g_dir.incident(v, mode="out"))
    g_dir.delete_edges(edges_to_delete)

    reset_0 = [0.0] * V_count
    reset_1 = [0.0] * V_count
    inv_0 = 1.0 / len(side_0)
    inv_1 = 1.0 / len(side_1)
    for v in side_0:
        reset_0[v] = inv_0
    for v in side_1:
        reset_1[v] = inv_1

    # Directed Personalized PageRank on the modified absorbing graph
    P0 = g_dir.personalized_pagerank(directed=True, reset=reset_0, damping=0.99)
    P1 = g_dir.personalized_pagerank(directed=True, reset=reset_1, damping=0.99)

    S0_X_plus = sum(P0[v] for v in top_0)
    S1_X_plus = sum(P1[v] for v in top_0)
    S0_Y_plus = sum(P0[v] for v in top_1)
    S1_Y_plus = sum(P1[v] for v in top_1)

    denom_X = 0.5 * (S0_X_plus + S1_X_plus)
    denom_Y = 0.5 * (S0_Y_plus + S1_Y_plus)

    if denom_X <= 0 or denom_Y <= 0:
        return 0.0

    P_XX = (0.5 * S0_X_plus) / denom_X
    P_YX = (0.5 * S1_X_plus) / denom_X
    P_YY = (0.5 * S1_Y_plus) / denom_Y
    P_XY = (0.5 * S0_Y_plus) / denom_Y

    return float((P_XX * P_YY) - (P_XY * P_YX))


def prepare_graph_data(graph_path: Path):
    G = nx.read_graphml(graph_path).to_undirected()
    giant_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(giant_nodes).copy()

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

    g_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)
    degrees = np.array(g_ig.degree(), dtype=float)
    total_volume = float(np.sum(degrees))

    valid_indices = []
    opinions = []
    for i, n in enumerate(nodes):
        score = G.nodes[n].get("url_bias_score")
        if score is not None:
            valid_indices.append(i)
            opinions.append(float(score))

    return (
        g_ig,
        edges,
        degrees,
        total_volume,
        np.array(valid_indices, dtype=int),
        np.array(opinions, dtype=float),
    )


def spectral_bisect_metagraph(comm_graph: nx.Graph):
    if comm_graph.number_of_nodes() < 2:
        return None

    if not nx.is_connected(comm_graph):
        largest_cc = max(nx.connected_components(comm_graph), key=len)
        G_sub = comm_graph.subgraph(largest_cc).copy()
    else:
        G_sub = comm_graph

    if G_sub.number_of_nodes() < 2:
        return None

    try:
        L = nx.normalized_laplacian_matrix(G_sub, weight="weight")
        _, eigenvectors = eigsh(L, k=2, which="SM", maxiter=5000)
        fiedler = eigenvectors[:, 1]
    except Exception:
        return None

    meta_side_0 = {node for node, val in zip(G_sub.nodes(), fiedler) if val < 0}
    meta_side_1 = {node for node, val in zip(G_sub.nodes(), fiedler) if val >= 0}

    if len(G_sub) < len(comm_graph):
        for node in set(comm_graph.nodes()) - set(G_sub.nodes()):
            if len(meta_side_0) <= len(meta_side_1):
                meta_side_0.add(node)
            else:
                meta_side_1.add(node)

    if not meta_side_0 or not meta_side_1:
        return None

    return meta_side_0, meta_side_1


def evaluate_single_run(
    g_ig: ig.Graph,
    edges: list,
    degrees: np.ndarray,
    total_volume: float,
    valid_indices: np.ndarray,
    opinions: np.ndarray,
    resolution: float,
    seed: int,
):
    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    n_comms = len(partition)
    if n_comms < 2:
        return None

    membership = partition.membership
    comm_graph = nx.Graph()
    comm_graph.add_nodes_from(range(n_comms))

    for u_idx, v_idx in edges:
        cu = membership[u_idx]
        cv = membership[v_idx]
        if cu != cv:
            if comm_graph.has_edge(cu, cv):
                comm_graph[cu][cv]["weight"] += 1.0
            else:
                comm_graph.add_edge(cu, cv, weight=1.0)

    res = spectral_bisect_metagraph(comm_graph)
    if res is None:
        return None
    meta_side_0, _ = res

    side_0_idx = [i for i in range(g_ig.vcount()) if membership[i] in meta_side_0]
    side_1_idx = [i for i in range(g_ig.vcount()) if membership[i] not in meta_side_0]
    node_sides = np.zeros(g_ig.vcount(), dtype=int)
    node_sides[side_1_idx] = 1

    # ----------------------------------------------------
    # 1. PURELY UNSUPERVISED TOPOLOGICAL METRICS (Option A)
    # ----------------------------------------------------
    n_total = g_ig.vcount()
    p0 = len(side_0_idx) / n_total
    p1 = len(side_1_idx) / n_total
    minority_ratio = min(p0, p1)
    balance_factor = 4.0 * p0 * p1  # 1.0 at 50/50 split, 0.0 at whisker cut

    # Normalized Cut (NCut) on the full graph
    cut_size = sum(
        data["weight"]
        for u, v, data in comm_graph.edges(data=True)
        if (u in meta_side_0) != (v in meta_side_0)
    )
    vol_0 = float(np.sum(degrees[side_0_idx]))
    vol_1 = total_volume - vol_0
    ncut = (cut_size / vol_0 + cut_size / vol_1) if (vol_0 > 0 and vol_1 > 0) else 2.0

    # Fast Corrected RWC (PageRank variant)
    rwc_score = calculate_rwc_rwr_fast(g_ig, degrees, side_0_idx, side_1_idx)

    # Unsupervised Topological Quality Score:
    # Rewards strong structural separation (RWC) while penalizing degenerate whisker cuts (balance_factor)
    topo_quality = rwc_score * balance_factor

    # ----------------------------------------------------
    # 2. DOWNSTREAM IDEOLOGICAL EVALUATION (Held-out check)
    # ----------------------------------------------------
    valid_sides = node_sides[valid_indices]
    if len(np.unique(valid_sides)) < 2:
        r_pb, p_val = 0.0, 1.0
    else:
        mean_0 = np.mean(opinions[valid_sides == 0])
        mean_1 = np.mean(opinions[valid_sides == 1])
        if mean_0 > mean_1:
            valid_sides = 1 - valid_sides
        r_pb, p_val = stats.pointbiserialr(valid_sides, opinions)

    return {
        "rwc": rwc_score,
        "minority_ratio": minority_ratio,
        "balance_factor": balance_factor,
        "ncut": ncut,
        "topo_quality": topo_quality,
        "r_pb": float(r_pb),  # type: ignore
        "p_value": float(p_val),  # type: ignore
        "n_communities": n_comms,
    }


def run_unsupervised_calibration():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for group_name, topic_list in TOPIC_GROUPS.items():
        for topic in topic_list:
            graph_path = get_graph_path(topic)
            if not graph_path.exists():
                print(f"[WARNING] Skipping '{topic}': {graph_path} not found")
                continue

            print(f"\nLoading '{topic}' [{group_name}]...")
            g_ig, edges, degrees, total_vol, valid_indices, opinions = (
                prepare_graph_data(graph_path)
            )

            total_iters = len(RESOLUTIONS) * N_SIMULATIONS
            with tqdm(total=total_iters, desc=f"Simulating {topic}") as pbar:
                for res in RESOLUTIONS:
                    for sim_idx in range(N_SIMULATIONS):
                        seed = 42 + sim_idx * 1000 + int(res * 100)
                        metrics = evaluate_single_run(
                            g_ig,
                            edges,
                            degrees,
                            total_vol,
                            valid_indices,
                            opinions,
                            resolution=float(res),
                            seed=seed,
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
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved calibration data to: {OUTPUT_CSV}")
    return df


def plot_unsupervised_calibration(df: pd.DataFrame):
    if df.empty:
        return

    # Find the purely unsupervised optimal resolution across political graphs
    pol_df = df[df["group"] == "Political"]
    agg_topo = pol_df.groupby("resolution")["topo_quality"].agg(["mean", "std"])
    agg_topo["penalized_score"] = agg_topo["mean"] - 0.5 * agg_topo["std"]
    best_gamma = agg_topo["penalized_score"].idxmax()
    print(
        f"\n>>> Purely Unsupervised Topological Optimal Resolution: gamma = {best_gamma} <<<"
    )

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    palette = {"Political": "#d95f02", "Non-Controversial (Control)": "#7570b3"}

    # --- Panel 1: Unsupervised Topological Objective (RWC x Balance) ---
    sns.lineplot(
        data=df,
        x="resolution",
        y="topo_quality",
        hue="group",
        palette=palette,
        linewidth=2.8,
        markers=True,
        errorbar="sd",
        ax=ax1,
    )
    ax1.axvline(
        best_gamma,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=rf"Topological Optimum ($\gamma = {best_gamma}$)",
    )
    ax1.axvline(
        0.3,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=r"Reference $\gamma = 0.3$",
    )
    ax1.set_title(
        r"Step 1 (Unsupervised Selection): Topological Partition Quality ($\mathrm{RWC} \times \mathrm{Balance}$)",
        fontweight="bold",
    )
    ax1.set_ylabel("Topological Score")
    ax1.legend(loc="upper right")

    # --- Panel 2: Partition Balance (Whisker-Cut Prevention) ---
    sns.lineplot(
        data=df,
        x="resolution",
        y="minority_ratio",
        hue="group",
        palette=palette,
        linewidth=2.5,
        markers=True,
        errorbar="sd",
        legend=False,
        ax=ax2,
    )
    ax2.axvline(best_gamma, color="crimson", linestyle="--", linewidth=1.8)
    ax2.axvline(0.3, color="black", linestyle=":", linewidth=1.5)
    ax2.set_title(
        "Step 2 (Unsupervised Diagnostic): Macroscopic Partition Balance (Avoiding Whisker Cuts)",
        fontweight="bold",
    )
    ax2.set_ylabel(r"Minority Ratio ($\min(|S_0|, |S_1|)/N$)")
    ax2.set_ylim(0, 0.55)

    # --- Panel 3: Downstream Ideological Validation (Point-Biserial r_pb) ---
    sns.lineplot(
        data=df,
        x="resolution",
        y="r_pb",
        hue="group",
        palette=palette,
        linewidth=2.8,
        markers=True,
        errorbar="sd",
        legend=False,
        ax=ax3,
    )
    ax3.axvline(best_gamma, color="crimson", linestyle="--", linewidth=1.8)
    ax3.axvline(0.3, color="black", linestyle=":", linewidth=1.5)
    ax3.set_title(
        r"Step 3 (Out-of-Loop Validation): Ideological Alignment ($r_{pb}$) at Selected $\gamma$",
        fontweight="bold",
    )
    ax3.set_xlabel(r"Leiden Resolution Parameter ($\gamma$)")
    ax3.set_ylabel("Point-Biserial Correlation ($r_{pb}$)")
    ax3.set_xticks(RESOLUTIONS)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"Saved 3-panel calibration plot to: {OUTPUT_PLOT}")
    plt.show()


if __name__ == "__main__":
    df_res = run_unsupervised_calibration()
    plot_unsupervised_calibration(df_res)
