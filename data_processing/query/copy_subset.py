import orjson
from data_processing.util.functions import (
    iter_jsonl,
    get_post_hashtags,
    extract_hashtags_from_jsonl,
)
from data_processing.config.logging import logger
from data_processing.config.paths import TOPICS_PATH

"""
This script is used to make subsequent queries for keywords (hashtags) from a single
.jsonl file (e.g., the output of copy_keyword_posts.py).

INPUT_SUBSET (.jsonl)
TARGET_HASHTAGS (List)
OUTPUT_FILE (.jsonl)
"""

INPUT_SUBSET = TOPICS_PATH / "_multiple/20260405/posts.jsonl"

# TARGET_HASHTAGS = ["#ukraine"]

path = TOPICS_PATH / "#aiethics/co_hashtags/chosen_closest_hashtags.jsonl"
TARGET_HASHTAGS = extract_hashtags_from_jsonl(path)

OUTPUT_FILE = TOPICS_PATH / "#aiethics/co_hashtags/posts_co_hashtags.jsonl"


def filter_subset():
    # Maintain consistency: lowercase and remove leading '#'
    clean_targets = {k.lower().lstrip("#") for k in TARGET_HASHTAGS}

    if not INPUT_SUBSET.exists():
        logger.error(f"Input subset not found at {INPUT_SUBSET}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    match_count = 0
    total_processed = 0

    logger.info(f"Filtering {INPUT_SUBSET.name} for hashtags: {TARGET_HASHTAGS}")

    with open(OUTPUT_FILE, "wb") as out_f:
        # We use your existing iter_jsonl for consistent I/O
        for post in iter_jsonl(INPUT_SUBSET):
            total_processed += 1

            post_tags = get_post_hashtags(post)

            # Check for intersection
            if any(tag in clean_targets for tag in post_tags):
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
