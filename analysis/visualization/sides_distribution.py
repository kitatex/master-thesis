import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from config.paths import TOPICS_PATH


TOPIC_NAME = "covid-19"

MIN_RP = 2
MIN_URL = 3
LEIDEN_RESOLUTION = 1
ALGORITHM = "leidenmeta"


FILE = f"{TOPIC_NAME}/graph/partition/{TOPIC_NAME}_minrp2_part_{ALGORITHM}{LEIDEN_RESOLUTION}.graphml"
INPUT_GRAPH = TOPICS_PATH / FILE

OUTPUT_PLOT = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/eda/{TOPIC_NAME}_sides_distribution_{ALGORITHM}{LEIDEN_RESOLUTION}.png"
)


def main():
    print("Loading graph...")

    G = nx.read_graphml(INPUT_GRAPH)

    side0 = []
    side1 = []

    for node, attrs in G.nodes(data=True):
        opinion = attrs.get("url_bias_score")
        side = attrs.get("side")

        if opinion is None or side is None:
            continue

        opinion = float(opinion)
        side = int(side)

        if side == 0:
            side0.append(opinion)
        elif side == 1:
            side1.append(opinion)

    print(f"Side 0 users: {len(side0)}")
    print(f"Side 1 users: {len(side1)}")

    np.linspace(-1.5, 1.5, 31)

    plt.figure(figsize=(12, 6))

    plt.hist(
        side0,
        bins="fd",  # type: ignore
        density=True,
        alpha=0.5,
        color="#E66100",
        label=f"Side 0 (n={len(side0)})",
    )

    plt.hist(
        side1,
        bins="fd",  # type: ignore
        density=True,
        alpha=0.5,
        color="#5D3A9B",
        label=f"Side 1 (n={len(side1)})",
    )

    plt.xlabel("Estimated Opinion")
    plt.ylabel("Density")
    plt.title(f"{TOPIC_NAME} - Opinion Distribution for both sides")

    plt.legend()
    plt.grid(alpha=0.3)

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)

    print(f"Saved plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
