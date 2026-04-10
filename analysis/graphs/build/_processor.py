import igraph as ig


def prune_graph(g: ig.Graph, k_val: int = 2) -> ig.Graph:
    """
    Removes low-degree noise using k-core and extracts the giant connected component.
    """
    # 1. Directly get the k-core subgraph
    # This replaces the manual 'induced_subgraph' call that was failing
    g_pruned = g.k_core(k_val)

    # 2. Extract the Weakly Connected Giant Component
    # This ensures we focus on the primary discourse cluster
    components = g_pruned.components(mode="weak")  # type: ignore
    g_giant = components.giant()

    return g_giant
