import igraph as ig


def partition_graph_spectral(graphml_path: str):
    print("Loading graph...")
    g = ig.Graph.Read_GraphML(graphml_path)

    # Force the network into exactly 2 communities using spectral bisection
    print("Running Spectral Bisection (k=2)...")
    communities = g.community_leading_eigenvector(clusters=2)

    # Extract the two sides based on membership
    membership = communities.membership
    g.vs["community"] = membership

    side_0 = [v.index for v in g.vs if v["community"] == 0]
    side_1 = [v.index for v in g.vs if v["community"] == 1]

    print(f"Partition 0 size: {len(side_0)}")
    print(f"Partition 1 size: {len(side_1)}")

    return g, side_0, side_1
