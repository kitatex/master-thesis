import igraph as ig
import matplotlib.pyplot as plt


def prepare_and_visualize(g_giant, partition):
    print("1. Preparing Node Attributes for Visualization...")

    # Calculate degree (how many connections each user has) to use for node size
    g_giant.vs["degree"] = g_giant.degree()

    # Assign community membership to a node attribute
    g_giant.vs["community"] = partition.membership

    # ---------------------------------------------------------
    print("2. Filtering for Top Communities...")
    # Get the IDs of the top 5 largest communities
    community_sizes = partition.sizes()
    top_5_community_ids = sorted(
        range(len(community_sizes)), key=lambda i: community_sizes[i], reverse=True
    )[:5]

    # Create a subgraph containing ONLY nodes from the top 5 communities
    # This removes the noise of the 20 tiny splinter groups
    top_nodes = [v.index for v in g_giant.vs if v["community"] in top_5_community_ids]
    g_top = g_giant.subgraph(top_nodes)

    print(f"Top Communities Subgraph: {g_top.vcount()} nodes, {g_top.ecount()} edges")

    # ---------------------------------------------------------
    print("3. Exporting to GraphML (For Gephi)...")
    # This is the file you will use for your final thesis visualizations
    g_top.write_graphml("climatecrisis_top_communities.graphml")
    print("Exported 'climatecrisis_top_communities.graphml'")

    # ---------------------------------------------------------
    print("4. Generating Quick Python Preview...")
    # Note: igraph plotting in python requires the 'cairo' or 'matplotlib' backend.

    # Assign distinct colors to the top 5 communities
    palette = ig.RainbowPalette(n=5)

    # Map the original community IDs to a 0-4 index for the palette
    community_to_color_index = {cid: idx for idx, cid in enumerate(top_5_community_ids)}

    colors = [palette.get(community_to_color_index[v["community"]]) for v in g_top.vs]
    g_top.vs["color"] = colors

    # Scale node sizes logarithmically based on degree so hubs are visible but not overwhelming
    import math

    g_top.vs["size"] = [math.log(d + 2) * 3 for d in g_top.vs["degree"]]

    # Use a force-directed layout (Fruchterman-Reingold)
    layout = g_top.layout_fruchterman_reingold()

    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ig.plot(
        g_top,
        target=ax,
        layout=layout,
        vertex_size=g_top.vs["size"],
        vertex_color=g_top.vs["color"],
        edge_width=0.2,
        edge_color="rgba(200, 200, 200, 0.5)",
        vertex_frame_width=0.1,
    )
    plt.title("Bluesky #climatecrisis: Top 5 Structural Communities")
    plt.show()


# --- Execution ---
# Assuming 'g_giant' and 'partition' are still in your active memory from the previous script:
# prepare_and_visualize(g_giant, partition)
