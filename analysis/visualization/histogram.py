import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from config.paths import TOPICS_PATH

# --- Configuration ---
TOPIC_NAME = "0 pokemon"
MIN_RP = 2
MIN_URL = 3
URL_SOURCE = "global"
INTERPOLATE = True

interp_str = "interpolated" if INTERPOLATE else "filtered"

INPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/with_opinions/{TOPIC_NAME}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}_{interp_str}.graphml"
)
OUTPUT_PLOT = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/eda/distribution_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}_{interp_str}.png"
)


def plot_opinion_distribution(graphml_path, output_plot, is_interpolated_run=True):
    print(f"Loading network from {graphml_path}...")
    G = nx.read_graphml(graphml_path)

    orig_opinions = []
    interp_opinions = []

    # Extract finalized opinions and metadata directly from the graph
    for node, data in G.nodes(data=True):
        score = data.get("url_bias_score")
        if score is None:
            continue

        # Handle GraphML boolean serialization
        is_interp = data.get("is_interpolated")
        if isinstance(is_interp, str):
            is_interp = is_interp.lower() == "true"

        if is_interp:
            interp_opinions.append(score)
        else:
            orig_opinions.append(score)

    print("Generating histogram...")
    fig, ax_hist = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("coolwarm")

    # Corrected normalization to prevent artificial right-wing saturation
    norm = mcolors.Normalize(vmin=-1.5, vmax=0.5, clip=True)

    bins = 30

    if is_interpolated_run:
        # plt.hist returns [Original_Patches, Interpolated_Patches] when stacked
        _, _, patches = ax_hist.hist(
            [orig_opinions, interp_opinions],
            bins=bins,
            range=(-1.5, 0.5),
            stacked=True,
            edgecolor="black",
            alpha=0.8,
        )

        orig_patches = patches[0]  # type: ignore
        interp_patches = patches[1]  # type: ignore

        # Apply colormap to Original data
        for patch in orig_patches:  # type: ignore
            x = patch.get_x() + patch.get_width() / 2
            patch.set_facecolor(cmap(norm(x)))

        # Apply colormap AND hatching to Interpolated data
        for patch in interp_patches:  # type: ignore
            x = patch.get_x() + patch.get_width() / 2
            patch.set_facecolor(cmap(norm(x)))
            patch.set_hatch("////")

        # Build custom proxy artists for an accurate legend
        legend_elements = [
            Patch(facecolor="lightgray", edgecolor="black", label="Original"),
            Patch(
                facecolor="lightgray",
                edgecolor="black",
                hatch="////",
                label="Interpolated",
            ),
        ]
        ax_hist.legend(handles=legend_elements, loc="upper right")

    else:
        _, _, patches = ax_hist.hist(
            orig_opinions, bins=bins, range=(-1.5, 0.5), edgecolor="black", alpha=0.8
        )
        for patch in patches:  # type: ignore
            x = patch.get_x() + patch.get_width() / 2
            patch.set_facecolor(cmap(norm(x)))

    ax_hist.set_title(
        f"Estimated User Opinion Distribution (Min {MIN_URL} URLs)",
        fontsize=14,
    )
    ax_hist.set_xlabel("Estimated Opinions (Left -1.5 to Right 0.5)", fontsize=12)
    ax_hist.set_ylabel("Number of Users", fontsize=12)
    ax_hist.set_xlim(-1.6, 0.6)
    ax_hist.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    os.makedirs(output_plot.parent, exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    print(f"Visualization saved to {output_plot}")


if __name__ == "__main__":
    plot_opinion_distribution(INPUT_GRAPHML, OUTPUT_PLOT, INTERPOLATE)
