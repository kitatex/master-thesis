import orjson
from pathlib import Path
from typing import List, Dict, Any

from query.config.paths import CURRENT_INPUT_PATH, CURRENT_OUTPUT_PATH
from query.config.logging import logger

"""Filter out any posts that are NOT in English."""


def filter_english_only_posts(input_file: Path, output_file: Path):
    if not input_file.exists():
        logger.error(f"Error: Input file not found at '{input_file}'")
        return

    english_posts: List[Dict[str, Any]] = []
    total_lines = 0
    skipped_count = 0

    logger.info(f"Reading posts from '{input_file}' (JSON Lines format)...")

    with open(input_file, "rb") as f:
        for line in f:
            total_lines += 1
            if not line.strip():
                continue

            try:
                post = orjson.loads(line)

                # Extract the 'langs' field
                # It might be None, missing, a list, or strict ['eng']
                langs = post.get("langs")

                # Strict check:
                # 1. langs must not be None
                # 2. langs must be a list
                # 3. The list must contain exactly one element: "eng"
                if isinstance(langs, list) and langs == ["eng"]:
                    english_posts.append(post)
                else:
                    skipped_count += 1

            except orjson.JSONDecodeError:
                logger.warning(
                    f"Could not decode JSON on line {total_lines}. Skipping."
                )

    logger.info(f"Processed {total_lines:,} lines.")
    logger.info(f"Retained {len(english_posts):,} strictly English posts.")
    logger.info(f"Filtered out {skipped_count:,} non-English or mixed-language posts.")

    try:
        logger.info(
            f"Writing {len(english_posts):,} posts to '{output_file}' in JSON Lines format..."
        )
        with open(output_file, "wb") as f:
            for post in english_posts:
                f.write(orjson.dumps(post))
                f.write(b"\n")
        logger.info(f"Successfully saved English posts to '{output_file}'")
    except IOError as e:
        logger.error(f"Error writing to output file: {e}")


if __name__ == "__main__":
    # You can update CURRENT_INPUT_PATH/CURRENT_OUTPUT_PATH in your config
    # or pass specific paths here if needed.
    filter_english_only_posts(CURRENT_INPUT_PATH, CURRENT_OUTPUT_PATH)
