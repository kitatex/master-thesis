import warnings
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
# Replace with your exact 6 topic folder names
TOPICS = [
    "capitalism",
    "climate change",
    "covid-19",
    "immigration",
    "war in gaza",
    "war in ukraine",
]

MIN_RP = 2
MIN_URL = 3
URL_SOURCE = "global"
# Adjust suffix depending on which graphs from 02_map_to_graph.py you use:
# e.g., "_interpolated.graphml", "_filtered.graphml", or ".graphml"
GRAPH_SUFFIX = "_interpolated.graphml"

# Grid Search Parameters
RESOLUTIONS = np.round(np.arange(0.1, 2.05, 0.1), 2)
N_SIMULATIONS = 30  # Number of stochastic Leiden runs per (topic, resolution)

OUTPUT_DIR = TOPICS_PATH / "_hyperparameter_tuning"
OUTPUT_CSV = OUTPUT_DIR / "leiden_resolution_tuning_results.csv"
OUTPUT_PLOT = OUTPUT_DIR / "leiden_resolution_tuning_curve.png"


# ==========================================
# CORE FUNCTIONS
# ==========================================
def get_graph_path(topic: str) -> Path:
    """Resolves the input GraphML path with a fallback if suffix varies."""
    base_dir = TOPICS_PATH / f"{topic}/graph/with_opinions"
    primary = (
        base_dir / f"{topic}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}{GRAPH_SUFFIX}"
    )
    if primary.exists():
        return primary
    # Fallback to non-suffixed filename (as used in partition/leiden_meta.py)
    fallback = base_dir / f"{topic}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.graphml"
    return fallback


def prepare_graph_data(graph_path: Path):
    """
    Loads the NetworkX graph once, extracts the giant component,
    and pre-computes the igraph object and opinion mapping for fast simulation.
    """
    G = nx.read_graphml(graph_path).to_undirected()

    # Extract Giant Component
    giant_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(giant_nodes).copy()

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

    g_ig = ig.Graph(n=len(nodes), edges=edges, directed=False)

    # Pre-extract valid opinion indices and values (using integer indices 0..N-1)
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
        np.array(valid_indices, dtype=int),
        np.array(opinions, dtype=float),
    )


def spectral_bisect_metagraph(comm_graph: nx.Graph):
    """Performs normalized spectral bisection on the Leiden community metagraph."""
    if comm_graph.number_of_nodes() < 2:
        return None

    try:
        L = nx.normalized_laplacian_matrix(comm_graph, weight="weight")
        # Compute the 2 smallest eigenvalues/vectors
        _, eigenvectors = eigsh(L, k=2, which="SM", maxiter=5000)
        fiedler = eigenvectors[:, 1]
    except Exception:
        return None

    meta_side_0 = {node for node, val in zip(comm_graph.nodes(), fiedler) if val < 0}
    meta_side_1 = {node for node, val in zip(comm_graph.nodes(), fiedler) if val >= 0}

    if not meta_side_0 or not meta_side_1:
        return None

    return meta_side_0, meta_side_1


def evaluate_single_run(
    g_ig: ig.Graph,
    edges: list,
    valid_indices: np.ndarray,
    opinions: np.ndarray,
    resolution: float,
    seed: int,
):
    """Runs one iteration of Leiden + Spectral Meta-Partition and returns metrics."""
    # 1. Run Leiden with explicit random seed
    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    n_comms = len(partition)
    if n_comms < 2:
        return None

    membership = partition.membership  # List of community IDs for each vertex 0..N-1

    # 2. Build weighted community metagraph
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

    # 3. Spectral partition of the metagraph
    res = spectral_bisect_metagraph(comm_graph)
    if res is None:
        return None
    meta_side_0, _ = res

    # 4. Map back to node level (1 if in meta_side_1, 0 if in meta_side_0)
    node_sides = np.array(
        [0 if membership[i] in meta_side_0 else 1 for i in range(g_ig.vcount())]
    )

    # Calculate structural balance (minority side ratio)
    side_1_count = np.sum(node_sides)
    side_0_count = len(node_sides) - side_1_count
    minority_ratio = min(side_0_count, side_1_count) / len(node_sides)

    # 5. Extract sides for nodes with valid opinion scores
    valid_sides = node_sides[valid_indices]

    # If all opinionated nodes landed on one side, correlation is undefined (0)
    if len(np.unique(valid_sides)) < 2:
        return {
            "abs_r_pb": 0.0,
            "r_pb_canonical": 0.0,
            "p_value": 1.0,
            "n_communities": n_comms,
            "minority_ratio": minority_ratio,
        }

    # Canonical orientation: ensure Side 0 always has the lower mean opinion
    mean_0 = np.mean(opinions[valid_sides == 0])
    mean_1 = np.mean(opinions[valid_sides == 1])
    if mean_0 > mean_1:
        valid_sides = 1 - valid_sides

    r_pb, p_val = stats.pointbiserialr(valid_sides, opinions)

    return {
        "abs_r_pb": abs(float(r_pb)),  # type: ignore
        "r_pb_canonical": float(r_pb),  # type: ignore
        "p_value": float(p_val),  # type: ignore
        "n_communities": n_comms,
        "minority_ratio": minority_ratio,
    }


# ==========================================
# MAIN EXECUTION & PLOTTING
# ==========================================
def run_grid_search():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for topic in TOPICS:
        graph_path = get_graph_path(topic)
        if not graph_path.exists():
            print(f"[WARNING] Skipping '{topic}': File not found at {graph_path}")
            continue

        print(f"\nLoading and preparing '{topic}' ({graph_path.name})...")
        g_ig, edges, valid_indices, opinions = prepare_graph_data(graph_path)
        print(
            f"  Nodes: {g_ig.vcount()}, Edges: {g_ig.ecount()}, Opinionated Nodes: {len(opinions)}"
        )

        total_iters = len(RESOLUTIONS) * N_SIMULATIONS
        with tqdm(total=total_iters, desc=f"Simulating {topic}") as pbar:
            for res in RESOLUTIONS:
                for sim_idx in range(N_SIMULATIONS):
                    seed = 42 + sim_idx * 1000 + int(res * 100)
                    metrics = evaluate_single_run(
                        g_ig,
                        edges,
                        valid_indices,
                        opinions,
                        resolution=float(res),
                        seed=seed,
                    )
                    if metrics is not None:
                        records.append(
                            {
                                "topic": topic,
                                "resolution": float(res),
                                "simulation": sim_idx,
                                **metrics,
                            }
                        )
                    pbar.update(1)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved raw simulation results to: {OUTPUT_CSV}")
    return df


def plot_results(df: pd.DataFrame):
    if df.empty:
        print("No data to plot.")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Panel 1: |r_pb| across Resolutions ---
    sns.lineplot(
        data=df,
        x="resolution",
        y="abs_r_pb",
        hue="topic",
        style="topic",
        markers=True,
        dashes=False,
        errorbar="sd",
        alpha=0.65,
        ax=ax1,
    )

    # Plot the Grand Mean across all topics in bold black
    sns.lineplot(
        data=df,
        x="resolution",
        y="abs_r_pb",
        color="black",
        linewidth=3.0,
        label="Grand Mean (All Topics)",
        errorbar=None,
        ax=ax1,
    )

    ax1.axvline(
        0.3,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=r"Selected $\gamma = 0.3$",
    )
    ax1.set_title(
        r"Hyperparameter Calibration: Leiden Resolution ($\gamma$) vs. Ideological Alignment",
        fontweight="bold",
    )
    ax1.set_ylabel("Absolute Point-Biserial Correlation ($|r_{pb}|$)")
    ax1.set_ylim(0, 1.0)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    # --- Panel 2: Partition Balance (Minority Ratio) ---
    sns.lineplot(
        data=df,
        x="resolution",
        y="minority_ratio",
        hue="topic",
        style="topic",
        markers=True,
        dashes=False,
        errorbar="sd",
        alpha=0.65,
        legend=False,
        ax=ax2,
    )
    sns.lineplot(
        data=df,
        x="resolution",
        y="minority_ratio",
        color="black",
        linewidth=3.0,
        errorbar=None,
        legend=False,
        ax=ax2,
    )

    ax2.axvline(0.3, color="crimson", linestyle="--", linewidth=1.8)
    ax2.set_title(
        r"Structural Partition Balance vs. Leiden Resolution ($\gamma$)",
        fontweight="bold",
    )
    ax2.set_xlabel(r"Leiden Resolution Parameter ($\gamma$)")
    ax2.set_ylabel(r"Minority Partition Ratio ($\min(|S_0|, |S_1|) / N$)")
    ax2.set_ylim(0, 0.55)
    ax2.set_xticks(RESOLUTIONS)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"Saved calibration plot to: {OUTPUT_PLOT}")
    plt.show()


if __name__ == "__main__":
    df_results = run_grid_search()
    plot_results(df_results)
