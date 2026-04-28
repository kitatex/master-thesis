import pandas as pd
import json
import re
import tldextract
import os

from config.paths import TOPICS_PATH, MBFC_CSV_PATH

TOPIC_NAME = "#gamedev"  # e.g., #climatecrisis

INPUT_DATA = (
    TOPICS_PATH / f"{TOPIC_NAME}/hashtag_corpus/posts_merged_deduplicated.jsonl"
)

OUTPUT_DIR = TOPICS_PATH / f"{TOPIC_NAME}/eda/url_eda.csv"


BIAS_MAP = {
    "left": -1.0,
    "left-center": -0.5,
    "neutral": 0.0,
    "right-center": 0.5,
    "right": 1.0,
}


def extract_domains(text):
    """
    Intelligently extracts root domains from text,
    ignoring truncated paths (e.g., /world/2024/m...)
    """
    if not isinstance(text, str):
        return []

    # Regex to find anything that looks like a URL/domain
    # Matches optional http(s)://, optional www., and the domain structure
    url_pattern = r"(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"
    matches = re.findall(url_pattern, text)

    domains = []
    for match in matches:
        # Use tldextract to reliably get the "registered domain"
        # e.g., "bbc.co.uk" from "www.bbc.co.uk/news..."
        extracted = tldextract.extract(match)
        root_domain = f"{extracted.domain}.{extracted.suffix}"
        if root_domain:
            domains.append(root_domain.lower())

    return domains


def process_data(jsonl_path, mbfc_csv_path, output_csv_path):
    # Load MBFC dataset
    print("Loading MBFC Dataset...")
    mbfc_df = pd.read_csv(mbfc_csv_path)

    # Map categorical bias to numerical scores
    mbfc_df["bias_score"] = mbfc_df["bias"].map(BIAS_MAP)

    # Create a fast lookup dictionary
    mbfc_lookup = mbfc_df.set_index("source")[
        ["bias_score", "factual_reporting"]
    ].to_dict("index")

    # Parse JSONL Data
    print("Parsing Bluesky Posts...")
    records = []
    total_urls_found = 0
    matched_urls = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            post = json.loads(line.strip())
            text = post.get("text", "")

            # Extract domains from the post text
            domains = extract_domains(text)

            for domain in domains:
                total_urls_found += 1

                # Check if the domain is in our MBFC lookup table
                if domain in mbfc_lookup:
                    matched_urls += 1
                    bias_score = mbfc_lookup[domain]["bias_score"]
                    factual = mbfc_lookup[domain]["factual_reporting"]
                else:
                    bias_score = None
                    factual = None

                # Store the flattened record
                records.append(
                    {
                        "post_id": post.get("post_id"),
                        "user_id": post.get("user_id"),
                        "domain": domain,
                        "bias_score": bias_score,
                        "factual_reporting": factual,
                    }
                )

    results_df = pd.DataFrame(records)

    # Safely create directories if they don't exist ---
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:  # Checks if a directory path was actually provided
        os.makedirs(output_dir, exist_ok=True)

    # Save the processed data
    results_df.to_csv(output_csv_path, index=False)

    # Print Summary Statistics
    print("\n" + "=" * 30)
    print("📋 SUMMARY STATISTICS")
    print("=" * 30)
    print(f"Total Posts Processed: {results_df['post_id'].nunique()}")
    print(f"Total URLs Extracted: {total_urls_found}")
    print(f"URLs Matched with MBFC: {matched_urls}")
    if total_urls_found > 0:
        match_rate = (matched_urls / total_urls_found) * 100
        print(f"Match Rate: {match_rate:.2f}%")
    print("=" * 30 + "\n")


if __name__ == "__main__":
    process_data(INPUT_DATA, MBFC_CSV_PATH, OUTPUT_DIR)
