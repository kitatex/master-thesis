import orjson
from pathlib import Path
from typing import List, Dict, Any

from config.logging import logger
from config.paths import TOPICS_PATH

"""
Deduplicates any posts in a .jsonl file.
"""

TOPIC_NAME = "cooking"  # e.g., #climatechange

DATA_LEVEL = "keywords"  # seed_hashtag or hashtag_corpus


INPUT_PATH = TOPICS_PATH / f"{TOPIC_NAME}/{DATA_LEVEL}/posts.jsonl"

OUTPUT_PATH = TOPICS_PATH / f"{TOPIC_NAME}/{DATA_LEVEL}/posts_deduplicated.jsonl"


def deduplicate_posts_by_id(
    input_file: Path, output_file: Path, id_key: str = "post_id"
):
    if not input_file.exists():
        logger.error(f"Error: Input file not found at '{input_file}'")
        return

    seen_ids = set()
    unique_posts: List[Dict[str, Any]] = []
    total_lines = 0

    logger.info(f"Reading posts from '{input_file}' (JSON Lines format)...")

    with open(input_file, "rb") as f:
        for line in f:
            total_lines += 1
            if not line.strip():
                continue

            try:
                post = orjson.loads(line)
                post_id = post.get(id_key)

                if post_id is not None and post_id not in seen_ids:
                    unique_posts.append(post)
                    seen_ids.add(post_id)
            except orjson.JSONDecodeError:
                logger.warning(
                    f"Could not decode JSON on line {total_lines}. Skipping."
                )

    logger.info(f"Processed {total_lines:,} lines.")
    logger.info(f"Found {len(unique_posts):,} unique posts.")
    logger.info(
        f"Removed {total_lines - len(unique_posts):,} duplicate or invalid posts."
    )

    try:
        logger.info(
            f"Writing {len(unique_posts):,} unique posts to '{output_file}' in JSON Lines format..."
        )
        # Write the unique posts to the output file in JSON Lines format
        with open(output_file, "wb") as f:
            for post in unique_posts:
                # Serialize each post object and add a newline character
                f.write(orjson.dumps(post))
                f.write(b"\n")
        logger.info(f"Successfully saved unique posts to '{output_file}'")
    except IOError as e:
        logger.error(f"Error writing to output file: {e}")


if __name__ == "__main__":
    deduplicate_posts_by_id(INPUT_PATH, OUTPUT_PATH)
