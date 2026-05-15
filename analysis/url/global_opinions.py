import os
import re
import orjson
import tldextract
import pandas as pd
import networkx as nx
from tqdm import tqdm

from config.paths import TOPICS_PATH, PATH_USER_POSTS, MBFC_CSV_PATH

TOPIC_NAME = "#aiethics"

INPUT_GRAPHML = TOPICS_PATH / f"{TOPIC_NAME}/graph/network_k2_large_comp.graphml"
OUTPUT_CSV = TOPICS_PATH / f"{TOPIC_NAME}/eda/global_user_opinions.csv"

# Minimum matched MBFC URLs required to calculate a valid score
MIN_URLS_SHARED = 2

# Bias mapping dictionary
BIAS_MAP = {
    "left": -1.0,
    "left-center": -0.5,
    "neutral": 0.0,
    "right-center": 0.5,
    "right": 1.0,
}


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
        # Handle the igraph double/float trap we fixed previously
        raw_id = node_attr.get("name", node_attr.get("v_name", node))
        try:
            clean_id = str(int(float(raw_id)))
            user_ids.add(clean_id)
        except (ValueError, TypeError):
            user_ids.add(str(raw_id).strip())

    print(f"Found {len(user_ids)} unique users in the network.")
    return user_ids


def build_mbfc_lookup(mbfc_csv_path):
    """Loads MBFC and creates a quick lookup dictionary for scores."""
    df = pd.read_csv(mbfc_csv_path)
    if "bias_score" not in df.columns:
        df["bias_score"] = df["bias"].map(BIAS_MAP)

    # Drop rows without a valid score
    df = df.dropna(subset=["bias_score"])
    return df.set_index("source")["bias_score"].to_dict()


def process_global_opinions():
    # 1. Initialization
    mbfc_lookup = build_mbfc_lookup(MBFC_CSV_PATH)
    relevant_users = get_relevant_users(INPUT_GRAPHML)

    records = []
    users_with_missing_files = 0

    print(f"Scanning full user histories for {len(relevant_users)} users...")

    # 2. Iterate through only the relevant users
    # Using tqdm to show a progress bar (this might take a few minutes)
    for user_id in tqdm(relevant_users):
        file_path = PATH_USER_POSTS / f"{user_id}.jsonl"

        if not file_path.exists():
            users_with_missing_files += 1
            continue

        matched_scores = []
        matched_domains = []

        # Open the user's specific history file
        with open(file_path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    # Faster JSON parsing
                    data = orjson.loads(line)
                    text = data.get("text", "")

                    if text:
                        domains = extract_domains(text)
                        for domain in domains:
                            if domain in mbfc_lookup:
                                matched_scores.append(mbfc_lookup[domain])
                                matched_domains.append(domain)
                except orjson.JSONDecodeError:
                    continue

        # 3. Validation & Aggregation
        if len(matched_scores) >= MIN_URLS_SHARED:
            avg_score = sum(matched_scores) / len(matched_scores)
            records.append(
                {
                    "user_id": user_id,
                    "estimated_opinion": avg_score,
                    # Store the list of URLs as a JSON string for transparency
                    "urls": orjson.dumps(matched_domains).decode("utf-8"),
                }
            )

    # 4. Save the new baseline database
    results_df = pd.DataFrame(records)

    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    # 5. Summary Printout
    print("\n" + "=" * 40)
    print("GLOBAL OPINION EXTRACTION COMPLETE")
    print("=" * 40)
    print(f"Total Network Users Searched: {len(relevant_users)}")
    print(f"Users Missing Local JSONL: {users_with_missing_files}")
    print(
        f"Users Meeting Minimum URL Threshold ({MIN_URLS_SHARED}+): {len(results_df)}"
    )
    print(f"Saved Database to: {OUTPUT_CSV}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    process_global_opinions()
