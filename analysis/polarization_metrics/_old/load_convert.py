import pandas as pd
import networkx as nx
from pathlib import Path

from config.paths import RWC_PATH

# Define your input and output directories
INPUT_DIRECTORY = RWC_PATH / "graphs_txt_selection"
OUTPUT_DIRECTORY = RWC_PATH / "graphs_graphml_selection"


def process_txt_to_graphml(input_txt: Path, output_graphml: Path):
    print(f"--- Processing {input_txt.name} ---")

    try:
        df = pd.read_csv(
            input_txt,
            header=None,
            names=["source", "target", "weight"],
            dtype={"source": str, "target": str, "weight": float},
        )

        df = df.dropna(subset=["source", "target"])

        # Preserve edge direction
        df_directed = df.groupby(["source", "target"], as_index=False)["weight"].sum()

        # Build directed graph
        g = nx.from_pandas_edgelist(
            df_directed,
            source="source",
            target="target",
            edge_attr="weight",
            create_using=nx.DiGraph(),  # type: ignore
        )  # type: ignore

        print(
            f"  Directed Graph: {g.number_of_nodes()} nodes, "
            f"{g.number_of_edges()} edges."
        )

        # Largest weakly connected component
        largest_wcc = max(nx.weakly_connected_components(g), key=len)

        g_giant = g.subgraph(largest_wcc).copy()

        print(
            f"  Giant WCC: {g_giant.number_of_nodes()} nodes, "
            f"{g_giant.number_of_edges()} edges."
        )

        nx.write_graphml(g_giant, output_graphml)

        print(f"  Saved to: {output_graphml.name}\n")

    except Exception as e:
        print(f"  [ERROR] Failed to process {input_txt.name}: {e}\n")


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Grab all .txt files in the input directory
    txt_files = list(INPUT_DIRECTORY.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {INPUT_DIRECTORY.resolve()}")
        return

    print(f"Found {len(txt_files)} network files. Starting conversion...\n")

    for txt_file in txt_files:
        # Save with a clear suffix so you know exactly what this file is
        output_filename = f"{txt_file.stem}_undir_gcc.graphml"
        output_filepath = OUTPUT_DIRECTORY / output_filename

        process_txt_to_graphml(txt_file, output_filepath)

    print("Batch conversion complete!")


if __name__ == "__main__":
    main()
