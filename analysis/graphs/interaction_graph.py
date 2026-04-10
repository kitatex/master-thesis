from pathlib import Path

from data_processing.config.paths import TOPICS_PATH, HT_CORPORA_PATH
from analysis.graphs.build._parser import extract_interactions
from analysis.graphs.build._constructor import create_weighted_df, df_to_igraph
from analysis.graphs.build._processor import prune_graph


"""
Executes the full graph building pipeline and exports both Parquet and GraphML.

INPUT_DATA (.jsonl): The data from which the graph is built.
OUTPUT_DIR: Directory where the graphs are saved.
"""

TOPIC_NAME = "#aiethics"  # e.g., #climatecrisis

INPUT_DATA = HT_CORPORA_PATH / f"{TOPIC_NAME}.jsonl"

OUTPUT_DIR = TOPICS_PATH / f"{TOPIC_NAME}/graph"

K_THRESHOLD = 2


def run_pipeline(
    input_jsonl: Path, output_dir: Path, topic: str, k_threshold: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting Build for {topic} ---")

    # Parse & Construct Weighted Table
    stream = extract_interactions(input_jsonl)
    weighted_df = create_weighted_df(stream)

    # Export to Parquet
    parquet_out = output_dir / f"{topic}_edges.parquet"
    weighted_df.to_parquet(parquet_out, index=False)
    print(f"Dataframe saved to: {parquet_out}")

    # Build & Prune Graph Object
    g = df_to_igraph(weighted_df)
    g_final = prune_graph(g, k_val=k_threshold)
    print(f"Graph pruned to {g_final.vcount()} nodes and {g_final.ecount()} edges.")

    # Export to GraphML
    graphml_out = output_dir / "network.graphml"
    g_final.write_graphml(str(graphml_out))
    print(f"GraphML saved to: {graphml_out}")

    print("Running Spectral Bisection (k=2)...")
    # Calculate communities
    communities = g_final.community_leading_eigenvector(clusters=2)
    # Assign the resulting sides (0 or 1) as a vertex attribute
    g_final.vs["side"] = communities.membership

    # Export to GraphML (this will automatically include the 'side' attribute)
    graphml_out = output_dir / "network_partitioned.graphml"
    g_final.write_graphml(str(graphml_out))
    print(f"Partitioned GraphML saved to: {graphml_out}")


if __name__ == "__main__":
    run_pipeline(INPUT_DATA, OUTPUT_DIR, TOPIC_NAME, K_THRESHOLD)
