import re
import orjson
import tldextract
import pandas as pd
from pathlib import Path

from config.logging import logger
from config.paths import TOPICS_PATH, MBFC_CSV_PATH

"""
Filters a .jsonl file to keep only posts that contain at least one valid URL
that exists in the MBFC (Media Bias/Fact Check) database.
"""

TOPIC_NAME = "defundpolice"

DATA_LEVEL = "full"  # seed_hashtag or hashtag_corpus or full

INPUT_PATH = TOPICS_PATH / f"{TOPIC_NAME}/{DATA_LEVEL}/posts.jsonl"
OUTPUT_PATH = TOPICS_PATH / f"{TOPIC_NAME}/{DATA_LEVEL}/posts_mbfc_matched.jsonl"


def build_mbfc_set(mbfc_csv_path: Path) -> set:
    """Loads MBFC and creates a fast lookup set of known domains."""
    df = pd.read_csv(mbfc_csv_path)

    # Extract the source column, drop empty rows, and ensure everything is lowercase
    sources = df["source"].dropna().astype(str).str.lower().unique()
    return set(sources)


def extract_domains(text: str) -> list:
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


def _has_mbfc_url(post_data: dict, mbfc_set: set) -> bool:
    """
    Check if the post text contains at least one URL present in the MBFC set.
    """
    text = post_data.get("text", "")
    if not text:
        return False

    domains = extract_domains(text)

    # Returns True the moment it finds a matching domain in the set
    return any(domain in mbfc_set for domain in domains)


def filter_mbfc_posts(input_file: Path, output_file: Path, mbfc_csv_path: Path):
    if not input_file.exists():
        logger.error(f"Error: Input file not found at '{input_file}'")
        return

    # 1. Build the lookup set first
    logger.info(f"Building MBFC lookup set from '{mbfc_csv_path}'...")
    try:
        mbfc_set = build_mbfc_set(mbfc_csv_path)
        logger.info(f"Successfully loaded {len(mbfc_set):,} unique domains from MBFC.")
    except Exception as e:
        logger.error(f"Failed to load MBFC lookup: {e}")
        return

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    match_count = 0

    logger.info(f"Reading posts from '{input_file}' (Filtering for MBFC URLs)...")

    try:
        # 2. Open files and stream
        with open(input_file, "rb") as in_f, open(output_file, "wb") as out_f:
            for line in in_f:
                total_lines += 1
                if not line.strip():
                    continue

                try:
                    post = orjson.loads(line)

                    # Pass the pre-loaded set to the checker
                    if isinstance(post, dict) and _has_mbfc_url(post, mbfc_set):
                        out_f.write(orjson.dumps(post) + b"\n")
                        match_count += 1

                except orjson.JSONDecodeError:
                    logger.warning(
                        f"Could not decode JSON on line {total_lines}. Skipping."
                    )

        logger.info(f"Processed {total_lines:,} lines.")
        logger.info(f"Found {match_count:,} posts with MBFC-matched URLs.")
        logger.info(f"Filtered out {total_lines - match_count:,} posts.")
        logger.info(f"Successfully saved MBFC posts to '{output_file}'")

    except IOError as e:
        logger.error(f"File I/O Error: {e}")


if __name__ == "__main__":
    filter_mbfc_posts(INPUT_PATH, OUTPUT_PATH, MBFC_CSV_PATH)
