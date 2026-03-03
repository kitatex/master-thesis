from pathlib import Path

from graphs.parser import extract_interactions
from graphs.constructor import create_weighted_df, df_to_igraph
from graphs.processor import prune_graph
from data_processing.config.paths import TOPICS_PATH


"""
Executes the full graph building pipeline and exports both Parquet and GraphML.

INPUT_DATA (.jsonl): The data from which the graph is built.
OUTPUT_DIR: Directory where the graphs are saved.
"""

TOPIC_NAME = "#climatecrisis"  # e.g., #climatecrisis

INPUT_DATA = TOPICS_PATH / f"{TOPIC_NAME}/hashtag_corpus/posts_deduplicated.jsonl"

OUTPUT_DIR = TOPICS_PATH / f"{TOPIC_NAME}/graph"

K_CORE_VALUE = 2


def run_build_pipeline(input_jsonl: Path, output_dir: Path, topic: str, k: int) -> None:
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting Build for {topic} ---")

    # 1. Parse & Construct Weighted Table
    stream = extract_interactions(input_jsonl)
    weighted_df = create_weighted_df(stream)

    # 2. Export to Parquet
    parquet_out = output_dir / f"{topic}_edges.parquet"
    weighted_df.to_parquet(parquet_out, index=False)
    print(f"Dataframe saved to: {parquet_out}")

    # 3. Build & Prune Graph Object
    g = df_to_igraph(weighted_df)
    g_final = prune_graph(g, k_val=k)
    print(f"Graph pruned to {g_final.vcount()} nodes and {g_final.ecount()} edges.")

    # 4. Export to GraphML
    graphml_out = output_dir / f"{topic}_network.graphml"
    g_final.write_graphml(str(graphml_out))
    print(f"GraphML saved to: {graphml_out}")


if __name__ == "__main__":
    run_build_pipeline(INPUT_DATA, OUTPUT_DIR, TOPIC_NAME, K_CORE_VALUE)
