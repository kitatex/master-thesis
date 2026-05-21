import igraph as ig
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection


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


def partition_graph_kernighan_lin(graphml_path: str):
    print("Loading graph...")
    g = ig.Graph.Read_GraphML(graphml_path)

    print("Extracting giant component...")
    g = g.components().giant()

    if g.is_directed():
        print("Converting directed graph to undirected...")
        g = g.as_undirected(combine_edges="sum")

    print("Converting to NetworkX graph...")
    nx_g = nx.Graph()

    nx_g.add_nodes_from(range(g.vcount()))
    nx_g.add_edges_from(g.get_edgelist())

    print("Running Kernighan-Lin bisection...")
    side_0_set, side_1_set = kernighan_lin_bisection(nx_g)

    side_0 = list(side_0_set)
    side_1 = list(side_1_set)

    membership = [0] * g.vcount()

    for node in side_1:
        membership[node] = 1

    g.vs["community"] = membership

    print(f"Partition 0 size: {len(side_0)}")
    print(f"Partition 1 size: {len(side_1)}")

    return g, side_0, side_1
