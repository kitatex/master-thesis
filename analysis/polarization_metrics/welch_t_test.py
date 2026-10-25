import networkx as nx
import numpy as np
from scipy import stats
from config.paths import TOPICS_PATH

try:
    from analysis.polarization_metrics.point_biserial import is_node_interpolated
except ImportError:
    from polarization_metrics.point_biserial import is_node_interpolated

# --- Configuration ---
TOPIC_NAME = "immigration"
MIN_RP = 2
ALGORITHM = "leidenmeta"
LEIDEN_RESOLUTION = 1.0

INPUT_GRAPH = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp{MIN_RP}_part_{ALGORITHM}{LEIDEN_RESOLUTION}.graphml"
)


def compute_welch_arrays(
    sides: np.ndarray, opinions: np.ndarray, canonical_orient: bool = True
) -> tuple[float, float, float]:
    """
    Computes Welch's t-test (unequal variances) between Side 0 and Side 1 opinions.
    If canonical_orient is True, Side 0 is oriented to have the lower mean opinion,
    and we return the positive t-statistic (Side 1 Mean - Side 0 Mean) / SE,
    the two-sided p-value, and the raw mean difference (mean_1 - mean_0).
    """
    if len(sides) < 4 or len(np.unique(sides)) < 2:
        return 0.0, 1.0, 0.0

    sides_arr = np.asarray(sides, dtype=int)
    opinions_arr = np.asarray(opinions, dtype=float)

    op_0 = opinions_arr[sides_arr == 0]
    op_1 = opinions_arr[sides_arr == 1]

    if len(op_0) < 2 or len(op_1) < 2:
        return 0.0, 1.0, 0.0

    mean_0 = float(np.mean(op_0))
    mean_1 = float(np.mean(op_1))

    # Canonically orient so Side 1 is the higher mean (making t_stat >= 0 and diff >= 0)
    if canonical_orient and mean_0 > mean_1:
        op_0, op_1 = op_1, op_0
        mean_0, mean_1 = mean_1, mean_0

    diff = mean_1 - mean_0
    t_stat, p_val = stats.ttest_ind(op_1, op_0, equal_var=False)

    if np.isnan(t_stat) or np.isnan(p_val):  # type: ignore
        return 0.0, 1.0, 0.0

    return float(t_stat), float(p_val), float(diff)  # type: ignore


def run_welch_ttest():
    print(f"Loading partitioned graph: {INPUT_GRAPH.name}...")
    G = nx.read_graphml(INPUT_GRAPH)

    all_sides, all_ops = [], []
    seed_sides, seed_ops = [], []

    for _, node_attr in G.nodes(data=True):
        if "side" in node_attr and "url_bias_score" in node_attr:
            s = int(node_attr["side"])
            v = float(node_attr["url_bias_score"])
            all_sides.append(s)
            all_ops.append(v)
            if not is_node_interpolated(node_attr):
                seed_sides.append(s)
                seed_ops.append(v)

    t_all, p_all, diff_all = compute_welch_arrays(
        np.array(all_sides), np.array(all_ops)
    )
    t_seed, p_seed, diff_seed = compute_welch_arrays(
        np.array(seed_sides), np.array(seed_ops)
    )

    print("\n" + "=" * 55)
    print("WELCH'S T-TEST RESULTS")
    print("=" * 55)
    print(
        f"[All Nodes (n={len(all_sides)})]  t = {t_all:.4f} | Δμ = {diff_all:.4f} | p = {p_all:.4e}"
    )
    if len(seed_sides) > 0:
        print(
            f"[Seed Only (n={len(seed_sides)})] t = {t_seed:.4f} | Δμ = {diff_seed:.4f} | p = {p_seed:.4e}"
        )
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_welch_ttest()
