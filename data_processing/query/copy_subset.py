import orjson
from data_processing.util.functions import (
    iter_jsonl,
    extract_keys_from_jsonl,
    extend_with_no_spaces,
)
from config.logging import logger
from config.paths import TOPICS_PATH

"""
This script is used to make subsequent queries for keywords (hashtags) from a single
.jsonl file (e.g., the output of copy_keyword_posts.py).

INPUT_SUBSET (.jsonl)
TARGET_HASHTAGS (List)
OUTPUT_FILE (.jsonl)
"""

INPUT_SUBSET = TOPICS_PATH / "_multiple/20261012/posts.jsonl"

TOPIC = "0 minecraft"

path = TOPICS_PATH / f"{TOPIC}/keywords/keywords.jsonl"
TARGET_KEYWORDS = extract_keys_from_jsonl(path)
clean_keywords = extend_with_no_spaces(TARGET_KEYWORDS)
print(f"{len(TARGET_KEYWORDS)} vs {len(clean_keywords)} keywords")

OUTPUT_FILE = TOPICS_PATH / f"{TOPIC}/keywords/posts.jsonl"


def filter_subset():
    clean_targets = [k.lower() for k in clean_keywords]

    if not INPUT_SUBSET.exists():
        logger.error(f"Input subset not found at {INPUT_SUBSET}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    match_count = 0
    total_processed = 0

    with open(OUTPUT_FILE, "wb") as out_f:
        for post in iter_jsonl(INPUT_SUBSET):
            total_processed += 1
            is_match = False

            post_text = post.get("text", "").lower()
            if post_text and any(kw in post_text for kw in clean_targets):
                is_match = True

            if is_match:
                out_f.write(orjson.dumps(post))
                out_f.write(b"\n")
                match_count += 1

            if total_processed % 100000 == 0:
                logger.info(f"Processed {total_processed:,} posts...")

    logger.info(
        f"Success! Found {match_count:,} matches out of {total_processed:,} posts."
    )
    logger.info(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    filter_subset()
