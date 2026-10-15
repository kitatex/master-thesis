import pandas as pd
import igraph as ig
import networkx as nx
import numpy as np

# from networkx.algorithms.community import kernighan_lin_bisection
from sklearn.cluster import SpectralClustering
from pathlib import Path

from config.paths import RWC_PATH

# The folder containing all the .txt files from the Garimella repo
INPUT_DIRECTORY = RWC_PATH / "graphs_txt_selection"
OUTPUT_DIRECTORY = RWC_PATH / "graphs_graphml_selection"


def process_garimella_txt(input_txt_path: Path, output_graphml_path: Path):
    print(f"--- Processing {input_txt_path.name} ---")

    try:
        # 1. Read the edge list
        # The Garimella format has no header: source, target, weight
        df = pd.read_csv(
            input_txt_path, header=None, names=["source", "target", "weight"]
        )

        # 2. Convert to igraph (Directed)
        g = ig.Graph.TupleList(df.itertuples(index=False), directed=True, weights=True)
        print(f"  Original graph: {g.vcount()} nodes, {g.ecount()} edges.")

        # 3. Extract the Giant Connected Component
        components = g.components(mode="weak")
        g_giant = components.giant()
        print(f"  Giant component: {g_giant.vcount()} nodes, {g_giant.ecount()} edges.")
        g_core = g_giant.k_core(2)
        print(f"  2-Core (Pruned): {g_core.vcount()} nodes, {g_core.ecount()} edges.")

        # 4. Partition the Graph (Kernighan-Lin Bisection)
        print("  Running Normalized Spectral Clustering (k=2)...")

        # Convert igraph -> NetworkX to get the SciPy sparse adjacency matrix
        nx_g = nx.Graph()
        nx_g.add_nodes_from(range(g_giant.vcount()))
        nx_g.add_edges_from(g_giant.get_edgelist())

        # Get the adjacency matrix and force it into CSR format
        adj_matrix = nx.to_scipy_sparse_array(nx_g, format="csr")

        # Cast the internal sparse matrix indices to 32-bit integers for scikit-learn
        adj_matrix.indices = adj_matrix.indices.astype(np.int32)
        adj_matrix.indptr = adj_matrix.indptr.astype(np.int32)

        # Run Spectral Clustering (Normalized Cut)
        # assign_labels='cluster_qr' is faster and more stable for graph partitions than kmeans
        sc = SpectralClustering(
            n_clusters=2, affinity="precomputed", assign_labels="cluster_qr"
        )
        membership = sc.fit_predict(adj_matrix)

        g_giant.vs["side"] = membership.tolist()

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
        output_filename = f"{txt_file.stem}_partitioned_k2.graphml"
        output_filepath = output_dir / output_filename

        process_garimella_txt(txt_file, output_filepath)

    print("Batch conversion complete!")


if __name__ == "__main__":
    process_directory(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
