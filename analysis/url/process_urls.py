import pandas as pd
import json
import re
import tldextract
import os

from config.paths import TOPICS_PATH, MBFC_CSV_PATH

TOPIC_NAME = "#aiethics"  # e.g., #climatecrisis

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
    # Load enriched MBFC dataset
    print("Loading Enriched MBFC Dataset...")
    mbfc_df = pd.read_csv(mbfc_csv_path)

    # Map categorical bias to numerical scores
    mbfc_df["bias_score"] = mbfc_df["bias"].map(BIAS_MAP)

    # Create an expanded lookup dictionary including new attributes
    meta_columns = [
        "bias_score",
        "factual_reporting",
        "country",
        "media_type",
        "popularity",
        "mbfc_credibility_rating",
    ]

    mbfc_lookup = mbfc_df.set_index("source")[meta_columns].to_dict("index")

    # Parse JSONL Data
    print("Parsing Bluesky Posts...")
    records = []
    total_urls_found = 0
    matched_urls = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            post = json.loads(line.strip())
            text = post.get("text", "")
            domains = extract_domains(text)

            for domain in domains:
                total_urls_found += 1
                if domain in mbfc_lookup:
                    matched_urls += 1
                    entry = mbfc_lookup[domain]

                    # Store the flattened record with all new attributes
                    records.append(
                        {
                            "post_id": post.get("post_id"),
                            "user_id": post.get("user_id"),
                            "domain": domain,
                            "bias_score": entry["bias_score"],
                            "factual_reporting": entry["factual_reporting"],
                            "country": entry["country"],
                            "media_type": entry["media_type"],
                            "popularity": entry["popularity"],
                            "credibility": entry["mbfc_credibility_rating"],
                        }
                    )
                else:
                    records.append(
                        {
                            "post_id": post.get("post_id"),
                            "user_id": post.get("user_id"),
                            "domain": domain,
                            "bias_score": None,
                            "factual_reporting": None,
                            "country": None,
                            "media_type": None,
                            "popularity": None,
                            "credibility": None,
                        }
                    )

    results_df = pd.DataFrame(records)
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(output_csv_path, index=False)

    print("\n" + "=" * 30)
    print("SUMMARY STATISTICS")
    print("=" * 30)
    print(f"Total Posts Processed: {results_df['post_id'].nunique()}")
    print(f"Total URLs Extracted: {total_urls_found}")
    print(f"Matched with Enriched MBFC: {matched_urls}")
    if total_urls_found > 0:
        print(f"Match Rate: {(matched_urls / total_urls_found) * 100:.2f}%")
    print("=" * 30 + "\n")


if __name__ == "__main__":
    process_data(INPUT_DATA, MBFC_CSV_PATH, OUTPUT_DIR)
