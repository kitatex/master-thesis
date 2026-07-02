from pathlib import Path

from config.paths import TOPICS_PATH
from analysis.construct_graph.build._parser import extract_interactions
from analysis.construct_graph.build._constructor import create_weighted_df, df_to_igraph
from analysis.construct_graph.build._processor import prune_graph

TOPIC_NAME = "#gamedev"
MIN_RP = 2
LARGEST_COMPONENT = True
DIRECTED = False
WEIGHTED = False

INPUT_DATA = TOPICS_PATH / f"{TOPIC_NAME}/full/posts.jsonl"
INPUT_DATA = TOPICS_PATH / f"{TOPIC_NAME}/full/posts_mbfc_matched.jsonl"

INPUT_DATA = TOPICS_PATH / f"{TOPIC_NAME}/keywords/posts.jsonl"

INPUT_DATA = TOPICS_PATH / f"{TOPIC_NAME}/full/posts_merged_deduplicated.jsonl"

OUTPUT_DIR = TOPICS_PATH / f"{TOPIC_NAME}/graph"

dir_string = "dir" if DIRECTED else "undir"
large_comp_string = "largecomp" if LARGEST_COMPONENT else "allcomp"
weight_string = "weighted" if WEIGHTED else "unweighted"

OUTPUT_GRAPH_RAW = (
    OUTPUT_DIR
    / f"{TOPIC_NAME}_minrp{MIN_RP}_{large_comp_string}_{dir_string}_{weight_string}.graphml"
)


def run_pipeline(input_jsonl: Path, topic: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting Build for {topic} ---")
    stream = extract_interactions(input_jsonl)
    weighted_df = create_weighted_df(stream)
    g = df_to_igraph(weighted_df)

    if not DIRECTED:
        print("Converting graph to undirected and combining edge weights...")
        g = g.as_undirected(mode="collapse", combine_edges="sum")

    print(f"Pruning graph (largest_comp={LARGEST_COMPONENT}, min_rp={MIN_RP})...")
    g_final = prune_graph(g, largest_comp=LARGEST_COMPONENT, min_edge_weight=MIN_RP)
    print(f"Graph pruned to {g_final.vcount()} nodes and {g_final.ecount()} edges.")

    if not WEIGHTED:
        print("WEIGHTED is False: Stripping edge weights before export...")
        if "weight" in g_final.edge_attributes():
            del g_final.es["weight"]

    g_final.write_graphml(str(OUTPUT_GRAPH_RAW))
    print(f"GraphML saved to: {OUTPUT_GRAPH_RAW}")


if __name__ == "__main__":
    run_pipeline(INPUT_DATA, TOPIC_NAME)
