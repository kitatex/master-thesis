import networkx as nx
import numpy as np
from scipy import stats
from config.paths import TOPICS_PATH

# --- Configuration ---
TOPIC_NAME = "immigration"
MIN_RP = 2
ALGORITHM = "leidenmeta"
LEIDEN_RESOLUTION = 0.3

INPUT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp{MIN_RP}_part_{ALGORITHM}{LEIDEN_RESOLUTION}.graphml"
)


def is_node_interpolated(node_attr: dict) -> bool:
    """Safely parses the 'is_interpolated' attribute from GraphML (bool or string)."""
    val = node_attr.get("is_interpolated", False)
    if isinstance(val, str):
        return val.strip().lower() == "true"
    return bool(val)


def compute_point_biserial_arrays(
    sides: np.ndarray, opinions: np.ndarray, canonical_orient: bool = True
) -> tuple[float, float]:
    """
    Computes Point-Biserial Correlation on numpy arrays.
    If canonical_orient is True, ensures Side 0 has the lower mean opinion.
    """
    if len(sides) < 2 or len(np.unique(sides)) < 2:
        return 0.0, 1.0

    sides_arr = np.asarray(sides, dtype=int)
    opinions_arr = np.asarray(opinions, dtype=float)

    if canonical_orient:
        mean_0 = np.mean(opinions_arr[sides_arr == 0])
        mean_1 = np.mean(opinions_arr[sides_arr == 1])
        if mean_0 > mean_1:
            sides_arr = 1 - sides_arr

    r_pb, p_value = stats.pointbiserialr(sides_arr, opinions_arr)
    if np.isnan(r_pb):  # type: ignore
        return 0.0, 1.0
    return float(r_pb), float(p_value)  # type: ignore


def compute_point_biserial_from_graph(
    G: nx.Graph, include_interpolated: bool = True
) -> tuple[float, float, int]:
    """
    Extracts 'side' and 'url_bias_score' from a partitioned NetworkX graph
    and returns (r_pb, p_value, n_users).
    """
    sides = []
    opinions = []

    for _, node_attr in G.nodes(data=True):
        if "side" in node_attr and "url_bias_score" in node_attr:
            if not include_interpolated and is_node_interpolated(node_attr):
                continue
            sides.append(int(node_attr["side"]))
            opinions.append(float(node_attr["url_bias_score"]))

    if not sides:
        return 0.0, 1.0, 0

    r_pb, p_val = compute_point_biserial_arrays(
        np.array(sides), np.array(opinions), canonical_orient=True
    )
    return r_pb, p_val, len(sides)


def run_point_biserial():
    print(f"Loading partitioned graph: {INPUT_GRAPH.name}...")
    G = nx.read_graphml(INPUT_GRAPH)

    r_all, p_all, n_all = compute_point_biserial_from_graph(
        G, include_interpolated=True
    )
    r_seed, p_seed, n_seed = compute_point_biserial_from_graph(
        G, include_interpolated=False
    )

    print("\n" + "=" * 50)
    print("POINT-BISERIAL CORRELATION RESULTS")
    print("=" * 50)
    print(f"[All Nodes (n={n_all})]      r_pb: {r_all:.4f} | p-value: {p_all:.4e}")
    if n_seed > 0 and n_seed != n_all:
        print(
            f"[Seed Nodes Only (n={n_seed})] r_pb: {r_seed:.4f} | p-value: {p_seed:.4e}"
        )
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_point_biserial()
