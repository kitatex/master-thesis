import orjson
import networkx as nx
from pathlib import Path

from config.paths import TOPICS_PATH
from config.logging import logger

"""
Extracts the final list of users from a processed GraphML file
and filters a posts.jsonl file to only include posts written by those users.
"""

TOPIC_NAME = "nuclearpower"
MIN_RP = 1
MIN_URL = 2
URL_SOURCE = "global"  # "global" or "discussion"

# --- Inputs ---
INPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.graphml"
)

# The large posts file you want to filter (e.g., your discussion posts)
INPUT_POSTS = TOPICS_PATH / f"{TOPIC_NAME}/full/posts.jsonl"

# --- Output ---
OUTPUT_POSTS = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/full/posts_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}_filtered.jsonl"
)


def get_graph_users(graphml_path: Path) -> set:
    """Extracts the unique, cleaned user IDs present in the final topic network."""
    logger.info(f"Loading network to extract final users: {graphml_path.name}")

    if not graphml_path.exists():
        logger.error(f"GraphML file not found: {graphml_path}")
        return set()

    G = nx.read_graphml(graphml_path)
    user_ids = set()

    for node in G.nodes():
        node_attr = G.nodes[node]
        # Handle the igraph double/float formatting trap
        raw_id = node_attr.get("name", node_attr.get("v_name", node))
        try:
            clean_id = str(int(float(raw_id)))
            user_ids.add(clean_id)
        except (ValueError, TypeError):
            user_ids.add(str(raw_id).strip())

    logger.info(f"Found {len(user_ids):,} unique users in the network.")
    return user_ids


def filter_posts_by_users(input_posts: Path, output_posts: Path, valid_users: set):
    """Streams the posts file and copies posts belonging to valid users."""
    if not valid_users:
        logger.error("No valid users provided. Aborting extraction.")
        return

    if not input_posts.exists():
        logger.error(f"Input posts file not found: {input_posts}")
        return

    output_posts.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    match_count = 0

    logger.info(f"Scanning {input_posts.name} for posts by network users...")

    try:
        with open(input_posts, "rb") as in_f, open(output_posts, "wb") as out_f:
            for line in in_f:
                total_lines += 1
                if not line.strip():
                    continue

                try:
                    # Parse JSON line
                    post = orjson.loads(line)

                    # Extract user ID and convert to string for safe set matching
                    u_id = str(post.get("user_id", "")).strip()

                    if u_id in valid_users:
                        out_f.write(
                            line
                        )  # We can write the raw bytes directly to save time!
                        match_count += 1

                except orjson.JSONDecodeError:
                    continue

                if total_lines % 500000 == 0:
                    logger.info(
                        f"Processed {total_lines:,} lines... Found {match_count:,} matches."
                    )

        logger.info("\n" + "=" * 40)
        print("POST EXTRACTION COMPLETE")
        print("=" * 40)
        logger.info(f"Total Posts Scanned: {total_lines:,}")
        logger.info(f"Posts Retained: {match_count:,}")
        logger.info(f"Saved Filtered Posts to: {output_posts}")
        print("=" * 40 + "\n")

    except IOError as e:
        logger.error(f"File I/O Error: {e}")


if __name__ == "__main__":
    # 1. Get the target users from the graph
    target_users = get_graph_users(INPUT_GRAPHML)

    # 2. Filter the posts file
    if target_users:
        filter_posts_by_users(INPUT_POSTS, OUTPUT_POSTS, target_users)
