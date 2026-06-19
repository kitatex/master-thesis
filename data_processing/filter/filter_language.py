import orjson
from pathlib import Path

from config.logging import logger
from config.paths import TOPICS_PATH

"""
Filters a .jsonl file to keep only strictly English posts based on their language tags.
"""

TOPIC_NAME = "immigration"

LANGUAGE = "eng"  # eng or deu (ISO 639-2 standard)
STRICT_MATCH = False  # whether exact match or whether multiple languages allowed

DATA_LEVEL = "keywords"  # "seed_hashtag" / "hashtag_corpus" / "full" / "keywords"

INPUT_PATH = TOPICS_PATH / f"{TOPIC_NAME}/{DATA_LEVEL}/posts.jsonl"

OUTPUT_PATH = TOPICS_PATH / f"{TOPIC_NAME}/{DATA_LEVEL}/posts_language_filter.jsonl"


def _is_language(post_data: dict, strict_match: bool) -> bool:
    langs = post_data.get("langs")
    if strict_match:
        return isinstance(langs, list) and langs == [LANGUAGE]
    else:
        return isinstance(langs, list) and LANGUAGE in langs


def filter_language_posts(input_file: Path, output_file: Path):
    if not input_file.exists():
        logger.error(f"Error: Input file not found at '{input_file}'")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    english_count = 0

    logger.info(f"Reading posts from '{input_file}' (Filtering for {LANGUAGE})...")

    try:
        # Open both files: read the input and stream matches straight to the output
        with open(input_file, "rb") as in_f, open(output_file, "wb") as out_f:
            for line in in_f:
                total_lines += 1
                if not line.strip():
                    continue

                try:
                    post = orjson.loads(line)

                    if isinstance(post, dict) and _is_language(post, STRICT_MATCH):
                        out_f.write(orjson.dumps(post) + b"\n")
                        english_count += 1

                except orjson.JSONDecodeError:
                    logger.warning(
                        f"Could not decode JSON on line {total_lines}. Skipping."
                    )

        logger.info(f"Processed {total_lines:,} lines.")
        logger.info(f"Found {english_count:,} {LANGUAGE} posts. Strict: {STRICT_MATCH}")
        logger.info(f"Filtered out {total_lines - english_count:,} posts.")
        logger.info(f"Successfully saved posts to '{output_file}'")

    except IOError as e:
        logger.error(f"File I/O Error: {e}")


if __name__ == "__main__":
    filter_language_posts(INPUT_PATH, OUTPUT_PATH)
