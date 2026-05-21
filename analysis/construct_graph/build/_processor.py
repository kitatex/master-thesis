import igraph as ig


def prune_graph(g: ig.Graph, largest_comp: bool, min_edge_weight: int) -> ig.Graph:
    """
    Removes weak connections using edge weights, cleans low-degree noise using k-core,
    and optionally extracts the giant connected component.
    """
    # Step 1: True Retweet Filtering (Filter by Edge Weight)
    if min_edge_weight > 1:
        # Select only edges where the retweet count is >= min_edge_weight
        # delete_vertices=False keeps nodes intact for the k-core step
        g = g.subgraph_edges(
            g.es.select(weight_ge=min_edge_weight), delete_vertices=False
        )

    # Step 2: Structural Pruning (k-core)
    # This now operates on the already-thinned network
    # g_pruned = g.k_core(k_val)

    # Step 3: Extract the Giant Component
    if largest_comp:
        components = g.components(mode="weak")  # type: ignore
        return components.giant()

    return g  # type: ignore
