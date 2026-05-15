import igraph as ig


def prune_graph(g: ig.Graph, largest_comp: bool, k_val: int = 2) -> ig.Graph:
    """
    Removes low-degree noise using k-core and optionally extracts the giant connected component.
    """
    # 1. Directly get the k-core subgraph
    g_pruned = g.k_core(k_val)

    # 2. Conditionally extract the Weakly Connected Giant Component
    if largest_comp:
        components = g_pruned.components(mode="weak")  # type: ignore
        return components.giant()

    # 3. If largest_comp is False, just return the k-core pruned graph
    return g_pruned  # type: ignore
