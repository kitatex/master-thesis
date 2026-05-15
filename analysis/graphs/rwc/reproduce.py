import pandas as pd
import igraph as ig
from pathlib import Path

from config.paths import RWC_PATH

# The folder containing all the .txt files from the Garimella repo
INPUT_DIRECTORY = RWC_PATH / "graphs_txt"

OUTPUT_DIRECTORY = RWC_PATH / "graphs_graphml"


def process_garimella_txt(input_txt_path: Path, output_graphml_path: Path):
    print(f"--- Processing {input_txt_path.name} ---")

    try:
        # 1. Read the edge list
        # The Garimella format has no header: source, target, weight
        df = pd.read_csv(
            input_txt_path, header=None, names=["source", "target", "weight"]
        )

        # 2. Convert to igraph
        g = ig.Graph.TupleList(df.itertuples(index=False), directed=True, weights=True)
        print(f"  Original graph: {g.vcount()} nodes, {g.ecount()} edges.")

        # 3. Extract the Giant Connected Component
        components = g.components(mode="weak")
        g_giant = components.giant()
        print(f"  Giant component: {g_giant.vcount()} nodes, {g_giant.ecount()} edges.")

        # 4. Partition the Graph (Spectral Bisection)
        communities = g_giant.community_leading_eigenvector(clusters=2)

        # Assign the resulting sides (0 or 1) as a vertex attribute
        g_giant.vs["side"] = communities.membership

        # Print partition sizes for validation
        side_0 = [v.index for v in g_giant.vs if v["side"] == 0]
        side_1 = [v.index for v in g_giant.vs if v["side"] == 1]
        print(f"  Partition 0 size: {len(side_0)}")
        print(f"  Partition 1 size: {len(side_1)}")

        # 5. Export to GraphML
        g_giant.write_graphml(str(output_graphml_path))
        print(f"  Saved to {output_graphml_path.name}\n")

    except Exception as e:
        print(f"  [ERROR] Failed to process {input_txt_path.name}: {e}\n")


def process_directory(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grab all .txt files in the input directory
    txt_files = list(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {input_dir.resolve()}")
        return

    print(f"Found {len(txt_files)} network files. Starting batch conversion...\n")

    for txt_file in txt_files:
        # Create output filename (e.g., netanyahu.txt -> netanyahu_partitioned_k2.graphml)
        output_filename = f"{txt_file.stem}_partitioned_k2.graphml"
        output_filepath = output_dir / output_filename

        process_garimella_txt(txt_file, output_filepath)

    print("Batch conversion complete!")


if __name__ == "__main__":
    process_directory(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
