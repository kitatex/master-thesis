import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx
from networkx.algorithms import bipartite
import warnings

from config.paths import TOPICS_PATH, MBFC_CSV_PATH

# --- Configuration & Paths ---
TOPIC_NAME = "#aiethics"
MIN_RP = 2
MIN_URL = 2

OPINIONS_CSV = TOPICS_PATH / f"{TOPIC_NAME}/eda/opinions_minrp2_minurl3.csv"
MBFC_CSV = MBFC_CSV_PATH

OUTPUT_DIR = TOPICS_PATH / f"{TOPIC_NAME}/eda/multidimensional/"

warnings.filterwarnings("ignore")


def load_and_preprocess(opinions_path, mbfc_path):
    # 1. Load data
    opinions = pd.read_csv(opinions_path)
    mbfc = pd.read_csv(mbfc_path)

    # Safely parse the stringified list of URLs
    opinions["urls"] = opinions["urls"].apply(ast.literal_eval)

    # Explode the dataframe so each user-url pair is a single row
    opinions_exploded = opinions.explode("urls").rename(columns={"urls": "source"})

    # Merge with MBFC metadata
    merged_df = opinions_exploded.merge(mbfc, on="source", how="inner")
    return opinions, merged_df


def plot_opinion_vs_credibility(merged_df):
    """Analyzes if there's a correlation between estimated opinion and media credibility."""
    # Group by user to get their average opinion and dominant credibility rating
    user_stats = (
        merged_df.groupby("user_id")
        .agg(
            {
                "estimated_opinion": "first",
                "mbfc_credibility_rating": lambda x: x.mode()[0]
                if not x.empty
                else "unknown",
            }
        )
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=user_stats, x="mbfc_credibility_rating", y="estimated_opinion")
    plt.title("Estimated Opinion vs. Primary Media Credibility")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def build_media_diet_matrix(merged_df):
    """Creates a matrix of users and the proportion of media types they consume."""
    diet_crosstab = pd.crosstab(
        merged_df["user_id"], merged_df["media_type"], normalize="index"
    )

    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    components = pca.fit_transform(diet_crosstab)

    diet_crosstab["PCA1"] = components[:, 0]
    diet_crosstab["PCA2"] = components[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=diet_crosstab, x="PCA1", y="PCA2", alpha=0.7)
    plt.title("User Media Diet Clusters (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.show()

    return diet_crosstab


def build_structural_network(merged_df):
    """Builds a User-User graph based on shared media sources for community detection."""
    B = nx.Graph()

    # Add nodes with bipartite attribute
    users = merged_df["user_id"].unique()
    sources = merged_df["source"].unique()

    B.add_nodes_from(users, bipartite=0)
    B.add_nodes_from(sources, bipartite=1)

    # Add edges
    edges = list(zip(merged_df["user_id"], merged_df["source"]))
    B.add_edges_from(edges)

    # Project to a User-User graph
    # Two users are connected if they share at least one media source
    user_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    user_graph = bipartite.projected_graph(B, user_nodes)

    print(
        f"User-User Structural Graph created with {user_graph.number_of_nodes()} nodes and {user_graph.number_of_edges()} edges."
    )
    return user_graph


if __name__ == "__main__":
    # Ensure you have 'opinions.csv' and 'mbfc.csv' in your working directory
    opinions_df, merged_df = load_and_preprocess(OPINIONS_CSV, MBFC_CSV)

    # Example execution (uncomment when running with real data):
    # plot_opinion_vs_credibility(merged_df)
    # media_matrix = build_media_diet_matrix(merged_df)
    structural_graph = build_structural_network(merged_df)
