from pathlib import Path
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection

from config.paths import TOPICS_PATH
from analysis.construct_graph.build._parser import extract_interactions
from analysis.construct_graph.build._constructor import create_weighted_df, df_to_igraph
from analysis.construct_graph.build._processor import prune_graph


"""
Executes the full graph building pipeline and exports both Parquet and GraphML.
"""

TOPIC_NAME = "#aiethics"
MIN_RP = 2
LARGEST_COMPONENT = True

INPUT_DATA = TOPICS_PATH / f"{TOPIC_NAME}/full/posts_merged_deduplicated.jsonl"

OUTPUT_DIR = TOPICS_PATH / f"{TOPIC_NAME}/graph"
large_comp_string = "largecomp" if LARGEST_COMPONENT else "allcomp"
OUTPUT_GRAPH_RAW = (
    OUTPUT_DIR / f"{TOPIC_NAME}_minrp{MIN_RP}_{large_comp_string}.graphml"
)
OUTPUT_GRAPH_PARTIT = OUTPUT_DIR / f"{TOPIC_NAME}_partit_minrp{MIN_RP}.graphml"


def run_pipeline(input_jsonl: Path, topic: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting Build for {topic} ---")

    # Parse & Construct Weighted Table
    stream = extract_interactions(input_jsonl)
    weighted_df = create_weighted_df(stream)

    # Export to Parquet
    # parquet_out = OUTPUT_DIR / f"{topic}_edges.parquet"
    # weighted_df.to_parquet(parquet_out, index=False)
    #  print(f"Dataframe saved to: {parquet_out}")

    # Build & Prune Graph Object
    g = df_to_igraph(weighted_df)
    if LARGEST_COMPONENT:
        g_final = prune_graph(g, LARGEST_COMPONENT, MIN_RP)
        print(f"Graph pruned to {g_final.vcount()} nodes and {g_final.ecount()} edges.")
    else:
        g_final = g
        print("No graph pruning")

    # Export to GraphML
    g_final.write_graphml(str(OUTPUT_GRAPH_RAW))
    print(f"GraphML saved to: {OUTPUT_GRAPH_RAW}")

    # print("Running Spectral Bisection (k=2)...")
    # communities = g_final.community_leading_eigenvector(clusters=2)
    # Assign the resulting sides (0 or 1) as a vertex attribute
    # g_final.vs["side"] = communities.membership

    print("Running Kernighan-Lin Bisection (k=2)...")

    # Convert igraph -> NetworkX
    nx_g = nx.Graph()
    nx_g.add_nodes_from(range(g_final.vcount()))
    nx_g.add_edges_from(g_final.get_edgelist())

    # Compute bipartition
    side_0, side_1 = kernighan_lin_bisection(nx_g)

    # Convert partition into membership vector
    membership = [0] * g_final.vcount()

    for node in side_1:
        membership[node] = 1

    # Assign the resulting sides (0 or 1) as a vertex attribute
    g_final.vs["side"] = membership
    g_final.write_graphml(str(OUTPUT_GRAPH_PARTIT))
    print(f"Partitioned GraphML saved to: {OUTPUT_GRAPH_PARTIT}")


if __name__ == "__main__":
    run_pipeline(INPUT_DATA, TOPIC_NAME)
