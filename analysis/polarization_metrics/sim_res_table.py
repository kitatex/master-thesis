import numpy as np
import pandas as pd
import networkx as nx
from config.paths import TOPICS_PATH


from analysis.partition.leiden_meta import (
    get_opinion_graph_path,
    extract_giant_component,
)
from analysis.polarization_metrics.point_biserial import is_node_interpolated


# ==========================================
# CONFIGURATION & PATHS
# ==========================================
N_SIMULATIONS = 1000
RESULTS_DIR = (TOPICS_PATH / "_polarization_results").resolve()
INPUT_CSV = RESULTS_DIR / f"polarization_simulations_n{N_SIMULATIONS}.csv"

OUTPUT_NUMERIC_CSV = (
    RESULTS_DIR / f"table_topic_summary_seed_n{N_SIMULATIONS}_numeric.csv"
)
OUTPUT_LATEX_TXT = RESULTS_DIR / f"table_topic_summary_seed_n{N_SIMULATIONS}_latex.tex"

ALPHA = 0.05  # Significance threshold for p-values

# Optional custom display names to match your other LaTeX tables
TOPIC_DISPLAY_MAP = {
    "bg3": "baldurs gate 3",
}


def clean_topic_name(raw_topic: str) -> str:
    """Removes leading '0 ' and applies any display name mappings."""
    cleaned = str(raw_topic).strip()
    if cleaned.startswith("0 "):
        cleaned = cleaned[2:].strip()
    cleaned = TOPIC_DISPLAY_MAP.get(cleaned, cleaned)
    return cleaned.replace("#", r"\#").replace("_", r"\_")


def get_seed_user_count(topic: str) -> int:
    """Helper to count non-interpolated seed users in a topic graph if needed for fallback."""
    graph_path = get_opinion_graph_path(topic, "global", 2, 3, True)
    if not graph_path.exists():
        return 0
    G = extract_giant_component(nx.read_graphml(graph_path))
    return sum(
        1
        for _, attr in G.nodes(data=True)
        if "url_bias_score" in attr and not is_node_interpolated(attr)
    )


def aggregate_topic_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the Monte Carlo runs into a 12-row DataFrame (one row per topic)
    using EXCLUSIVELY ground-truth seed user opinions for OPC (r_pb) and Welch's t-test.
    """
    # Fallback if the CSV was generated before welch_t_seed_only was explicitly logged
    has_welch_seed = (
        "welch_t_seed_only" in df.columns and "welch_p_seed_only" in df.columns
    )

    rows = []
    for (group, topic), sub in df.groupby(["group", "topic"], sort=False):
        n_runs = len(sub)

        r_seed_series = sub["r_pb_seed_only"]
        p_seed_series = sub["p_value_seed_only"]

        if has_welch_seed:
            t_seed_series = sub["welch_t_seed_only"]
            wp_seed_series = sub["welch_p_seed_only"]
        else:
            # Fallback: compute seed t-stat from r_pb_seed_only and seed user count
            n_seeds = get_seed_user_count(topic)  # type: ignore
            dof = max(1, n_seeds - 2)
            t_seed_series = r_seed_series * np.sqrt(
                dof / np.maximum(1e-12, 1.0 - r_seed_series**2)
            )
            wp_seed_series = p_seed_series

        opc_sig_pct = float((p_seed_series < ALPHA).mean() * 100.0)
        welch_sig_pct = float((wp_seed_series < ALPHA).mean() * 100.0)

        rows.append(
            {
                "group": group,
                "topic_raw": topic,
                "topic_display": clean_topic_name(topic),  # type: ignore
                "n_runs": n_runs,
                # Assortativity Coefficient (AC)
                "ac_mean": float(sub["ac"].mean()),
                "ac_std": float(sub["ac"].std()),
                # Adaptive EI Index (AEI)
                "aei_mean": float(sub["aei"].mean()),
                "aei_std": float(sub["aei"].std()),
                # Random Walk Controversy (RWC)
                "rwc_mean": float(sub["rwc"].mean()),
                "rwc_std": float(sub["rwc"].std()),
                # Seed-Only Opinion-Structural Point-Biserial Correlation (OPC / r_pb)
                "r_pb_seed_mean": float(r_seed_series.mean()),
                "r_pb_seed_std": float(r_seed_series.std()),
                "r_pb_seed_p_median": float(p_seed_series.median()),
                "r_pb_seed_sig_pct": opc_sig_pct,
                # Seed-Only Welch's t-test
                "welch_t_seed_mean": float(t_seed_series.mean()),
                "welch_t_seed_std": float(t_seed_series.std()),
                "welch_p_seed_median": float(wp_seed_series.median()),
                "welch_seed_sig_pct": welch_sig_pct,
            }
        )

    summary_df = pd.DataFrame(rows)

    # Sort topics alphabetically within Political and Non-Political groups to match tab:url_statistics
    pol_mask = summary_df["group"] == "Political"
    df_pol = summary_df[pol_mask].sort_values("topic_display")
    df_nonpol = summary_df[~pol_mask].sort_values("topic_display")

    return pd.concat([df_pol, df_nonpol], ignore_index=True)


def generate_latex_table(summary_df: pd.DataFrame) -> str:
    """
    Formats the 12-row summary table into the exact LaTeX style of tab:url_statistics,
    using \multicolumn section headers for Political and Non-Political Topics,
    and scales it to fit the \textwidth.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \caption{Polarization measures across political and non-political topics over $N = "
        + str(N_SIMULATIONS)
        + r"$ Monte Carlo simulations (Mean $\pm$ SD). Opinion-based metrics (OPC and Welch's $t$) are evaluated exclusively on non-interpolated seed users.}",
        r"    \label{tab:polarization_results}",
        r"    % Use \small to prevent aggressive scaling distortion",
        r"    \small",
        r"    \setlength{\tabcolsep}{5pt} % Tightened from 8pt to save horizontal space",
        r"    \resizebox{\textwidth}{!}{% Scales the tabular environment to exact text width",
        r"    \begin{tabular}{l r r r r r r}",
        r"        \toprule",
        r"        \textbf{Topic} & \textbf{AC} & \textbf{AEI} & \textbf{RWC} & \makecell[r]{\textbf{OPC}\\\textbf{($r_{pb}$)}} & \makecell[r]{\textbf{Welch's}\\\textbf{$t$-stat.}} & \makecell[r]{\textbf{\% Sig.}\\\textbf{($p < 0.05$)}} \\",
    ]

    pol_df = summary_df[summary_df["group"] == "Political"]
    nonpol_df = summary_df[summary_df["group"] != "Political"]

    sections = [
        ("Political Topics", pol_df),
        ("Non-Political Topics", nonpol_df),
    ]

    for section_title, sec_df in sections:
        if sec_df.empty:
            continue
        lines.append(r"        \midrule")
        lines.append(
            rf"        \multicolumn{{7}}{{@{{}}l}}{{\textit{{{section_title}}}}} \\"
        )
        lines.append(r"        \midrule")

        for _, r in sec_df.iterrows():
            t_name = f"{r['topic_display']:<16}"
            ac_str = f"${r['ac_mean']:.3f} \\pm {r['ac_std']:.3f}$"
            aei_str = f"${r['aei_mean']:.3f} \\pm {r['aei_std']:.3f}$"
            rwc_str = f"${r['rwc_mean']:.3f} \\pm {r['rwc_std']:.3f}$"
            opc_str = f"${r['r_pb_seed_mean']:.3f} \\pm {r['r_pb_seed_std']:.3f}$"
            welch_str = (
                f"${r['welch_t_seed_mean']:.2f} \\pm {r['welch_t_seed_std']:.2f}$"
            )
            sig_str = f"{r['welch_seed_sig_pct']:05.2f}"

            lines.append(
                f"        {t_name} & {ac_str} & {aei_str} & {rwc_str} & {opc_str} & {welch_str} & {sig_str} \\\\"
            )

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    } % End of resizebox",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main():
    if not INPUT_CSV.exists():
        fallback = RESULTS_DIR / "polarization_simulations_n100.csv"
        if fallback.exists():
            print(f"{INPUT_CSV.name} not found yet; loading {fallback.name} instead...")
            df = pd.read_csv(fallback)
        else:
            print(f"Error: Could not find {INPUT_CSV}")
            return
    else:
        print(f"Loading {INPUT_CSV}...")
        df = pd.read_csv(INPUT_CSV)

    summary_df = aggregate_topic_results(df)

    # 1. Save numeric CSV with _mean and _std columns
    with open(OUTPUT_NUMERIC_CSV, "w", encoding="utf-8", newline="") as f:
        summary_df.to_csv(f, index=False)
    print(f"Saved numeric summary CSV to:\n{OUTPUT_NUMERIC_CSV}")

    # 2. Save formatted LaTeX table
    latex_code = generate_latex_table(summary_df)
    with open(OUTPUT_LATEX_TXT, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"Saved LaTeX table to:\n{OUTPUT_LATEX_TXT}\n")

    # 3. Print preview in terminal
    print(latex_code)


if __name__ == "__main__":
    main()
