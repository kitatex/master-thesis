import pandas as pd
import igraph as ig


PARQUET_FILE = "C:/Users/leond/Documents/02-REPOS/06 master thesis/master-thesis/datasets/topics/#climatecrisis/graph/climatecrisis_edgelist.parquet"


def analyze_structural_network(parquet_path):
    print("1. Loading Data & Building Graph...")
    # Load the edgelist
    edges_df = pd.read_parquet(parquet_path)

    # Create directed graph from DataFrame
    # igraph automatically creates nodes from the 'source' and 'target' columns
    g = ig.Graph.TupleList(
        edges_df.itertuples(index=False), directed=True, weights=True
    )

    print(f"Original Graph: {g.vcount()} nodes, {g.ecount()} edges")

    # ---------------------------------------------------------
    print("\n2. Pruning the Graph (k-core decomposition)...")
    # Convert to undirected temporarily just for strict degree counting if you want,
    # but k_core natively handles total degree.
    # Let's drop nodes that don't have at least 2 connections in the network.
    g_pruned = g.k_core(2)

    print(f"Pruned Graph (k>=2): {g_pruned.vcount()} nodes, {g_pruned.ecount()} edges")

    # Get the giant connected component (drop isolated fragmented islands)
    components = g_pruned.components(mode="weak")
    g_giant = components.giant()

    print(f"Giant Component: {g_giant.vcount()} nodes, {g_giant.ecount()} edges")

    # ---------------------------------------------------------
    print("\n3. Community Detection (Leiden Algorithm)...")
    # For community detection, it is often mathematically cleaner to treat the graph as undirected
    g_undirected = g_giant.as_undirected(mode="collapse", combine_edges="sum")

    # Run the Leiden algorithm (optimizing Modularity)
    # Note: 'weights' tells the algorithm to respect how many times users interacted
    partition = g_undirected.community_leiden(
        objective_function="modularity", weights="weight"
    )

    # ---------------------------------------------------------
    print("\n4. Results & Modularity...")
    modularity_score = partition.modularity
    num_communities = len(partition)

    print(f"Modularity Score (Q): {modularity_score:.4f}")
    print(f"Number of structural communities found: {num_communities}")

    # Let's look at the sizes of the top 5 largest communities
    community_sizes = partition.sizes()
    community_sizes.sort(reverse=True)
    print(f"Top 5 Community Sizes: {community_sizes[:5]}")

    # Save the community assignments back to the nodes for later semantic mapping
    g_giant.vs["community"] = partition.membership

    return g_giant, partition


if __name__ == "__main__":
    g_final, part = analyze_structural_network(PARQUET_FILE)
