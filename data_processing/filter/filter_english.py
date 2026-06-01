import orjson
from pathlib import Path

from data_processing.util.functions import (
    iter_jsonl,
)  # Optional, but using explicit line read to match your script structure
from config.logging import logger
from config.paths import TOPICS_PATH

"""
Filters a .jsonl file to keep only strictly English posts based on their language tags.
"""

TOPIC_NAME = "nuclearpower"

DATA_LEVEL = "full"  # seed_hashtag or hashtag_corpus or full

INPUT_PATH = TOPICS_PATH / f"{TOPIC_NAME}/{DATA_LEVEL}/posts_merged_deduplicated.jsonl"

OUTPUT_PATH = TOPICS_PATH / f"{TOPIC_NAME}/{DATA_LEVEL}/posts_english.jsonl"


def _is_english(post_data: dict) -> bool:
    """
    Standardized check for strictly English posts.
    """
    langs = post_data.get("langs")
    return isinstance(langs, list) and langs == ["eng"]


def filter_english_posts(input_file: Path, output_file: Path):
    if not input_file.exists():
        logger.error(f"Error: Input file not found at '{input_file}'")
        return

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    english_count = 0

    logger.info(f"Reading posts from '{input_file}' (Filtering for English)...")

    try:
        # Open both files: read the input and stream matches straight to the output
        with open(input_file, "rb") as in_f, open(output_file, "wb") as out_f:
            for line in in_f:
                total_lines += 1
                if not line.strip():
                    continue

                try:
                    post = orjson.loads(line)

                    if isinstance(post, dict) and _is_english(post):
                        out_f.write(orjson.dumps(post) + b"\n")
                        english_count += 1

                except orjson.JSONDecodeError:
                    logger.warning(
                        f"Could not decode JSON on line {total_lines}. Skipping."
                    )

        logger.info(f"Processed {total_lines:,} lines.")
        logger.info(f"Found {english_count:,} strictly English posts.")
        logger.info(
            f"Filtered out {total_lines - english_count:,} non-English or invalid posts."
        )
        logger.info(f"Successfully saved English posts to '{output_file}'")

    except IOError as e:
        logger.error(f"File I/O Error: {e}")


if __name__ == "__main__":
    filter_english_posts(INPUT_PATH, OUTPUT_PATH)
