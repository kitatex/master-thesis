import os
import re
import orjson
import tldextract
import pandas as pd
import networkx as nx
from tqdm import tqdm

from config.paths import TOPICS_PATH, PATH_USER_POSTS

"""Get a global-level and discussion-level list of shared URLs for each user in a graph."""

TOPIC_NAME = "#gaza"
MIN_RP = 2  # has to match graph

INPUT_GRAPHML = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/graph/{TOPIC_NAME}_minrp{MIN_RP}_largecomp_undir_unweighted.graphml"
)

DISCUSSION_POSTS_FILE = TOPICS_PATH / f"{TOPIC_NAME}/full/posts_mbfc_matched.jsonl"
DISCUSSION_POSTS_FILE = TOPICS_PATH / f"{TOPIC_NAME}/keywords/posts.jsonl"

OUTPUT_CSV = TOPICS_PATH / f"{TOPIC_NAME}/metadata/extracted_urls_minrp{MIN_RP}.csv"


def extract_domains(text):
    """Extracts root domains intelligently, ignoring truncated paths."""
    if not isinstance(text, str):
        return []
    url_pattern = r"(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"
    matches = re.findall(url_pattern, text)
    domains = []
    for match in matches:
        extracted = tldextract.extract(match)
        root_domain = f"{extracted.domain}.{extracted.suffix}"
        if root_domain:
            domains.append(root_domain.lower())
    return domains


def get_relevant_users(graphml_path):
    """Extracts the unique, cleaned user IDs present in the topic network."""
    print(f"Loading network to extract relevant users: {graphml_path}")
    G = nx.read_graphml(graphml_path)
    user_ids = set()

    for node in G.nodes():
        node_attr = G.nodes[node]
        raw_id = node_attr.get("name", node_attr.get("v_name", node))
        try:
            clean_id = str(int(float(raw_id)))
            user_ids.add(clean_id)
        except (ValueError, TypeError):
            user_ids.add(str(raw_id).strip())

    print(f"Found {len(user_ids)} unique users in the network.")
    return user_ids


def process_extraction():
    relevant_users = get_relevant_users(INPUT_GRAPHML)

    # Initialize a dictionary to store lists of URLs for each user
    user_data = {uid: {"global": [], "discussion": []} for uid in relevant_users}

    # 1. Extract from the Discussion file
    print(f"Scanning discussion posts: {DISCUSSION_POSTS_FILE}")
    if DISCUSSION_POSTS_FILE.exists():
        with open(DISCUSSION_POSTS_FILE, "rb") as f:
            for line in tqdm(f, desc="Discussion Posts"):
                if not line.strip():
                    continue
                try:
                    data = orjson.loads(line)
                    u_id = str(data.get("user_id", ""))

                    if u_id in user_data:
                        text = data.get("text", "")
                        if text:
                            domains = extract_domains(text)
                            user_data[u_id]["discussion"].extend(domains)
                except orjson.JSONDecodeError:
                    continue
    else:
        print(f"Discussion file not found at {DISCUSSION_POSTS_FILE}. Skipping.")

    # 2. Extract from Global user history
    print(f"Scanning global user histories for {len(relevant_users)} users...")
    for user_id in tqdm(relevant_users, desc="Global History"):
        file_path = PATH_USER_POSTS / f"{user_id}.jsonl"

        if not file_path.exists():
            continue

        with open(file_path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = orjson.loads(line)
                    text = data.get("text", "")
                    if text:
                        domains = extract_domains(text)
                        user_data[user_id]["global"].extend(domains)
                except orjson.JSONDecodeError:
                    continue

    # 3. Format and save
    records = []
    for uid, urls in user_data.items():
        records.append(
            {
                "user_id": uid,
                "global_urls": orjson.dumps(urls["global"]).decode("utf-8"),
                "discussion_urls": orjson.dumps(urls["discussion"]).decode("utf-8"),
            }
        )

    results_df = pd.DataFrame(records)

    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 40)
    print("URL EXTRACTION COMPLETE")
    print("=" * 40)
    print(f"Saved Database to: {OUTPUT_CSV}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    process_extraction()
