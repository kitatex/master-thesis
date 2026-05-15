import orjson
from data_processing.util.functions import (
    iter_jsonl,
    get_post_hashtags,
    extract_keys_from_jsonl,
    extract_hashtags_from_jsonl,
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

INPUT_SUBSET = TOPICS_PATH / "_multiple/20260514/posts.jsonl"

TOPIC = "#gaza"

HASHTAG_ONLY = False

# --- HASHTAGS ---
# path = TOPICS_PATH / f"{TOPIC}/co_hashtags/chosen_closest_hashtags.jsonl"
# TARGET_KEYWORDS = extract_hashtags_from_jsonl(path)
# OUTPUT_FILE = TOPICS_PATH / f"{TOPIC}/co_hashtags/posts_co_hashtags.jsonl"

# --- KEYWORDS ---
path = TOPICS_PATH / f"{TOPIC}/keywords/chosen_keywords_bigrams.jsonl"
TARGET_KEYWORDS = extract_keys_from_jsonl(path)
OUTPUT_FILE = TOPICS_PATH / f"{TOPIC}/keywords/posts.jsonl"


def filter_subset():
    # 1. Setup clean targets based on the matching mode
    if HASHTAG_ONLY:
        # Maintain consistency: lowercase and remove leading '#' for tag intersection
        clean_targets = {k.lower().lstrip("#") for k in TARGET_KEYWORDS}
    else:
        # Lowercase for general string matching (keeps spaces for bigrams)
        clean_targets = [k.lower() for k in TARGET_KEYWORDS]

    if not INPUT_SUBSET.exists():
        logger.error(f"Input subset not found at {INPUT_SUBSET}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    match_count = 0
    total_processed = 0
    mode_label = "hashtags" if HASHTAG_ONLY else "general keywords"

    logger.info(f"Filtering {INPUT_SUBSET.name} for {mode_label}: {TARGET_KEYWORDS}")

    with open(OUTPUT_FILE, "wb") as out_f:
        for post in iter_jsonl(INPUT_SUBSET):
            total_processed += 1
            is_match = False

            if HASHTAG_ONLY:
                # Mode A: Check extracted hashtags only
                post_tags = get_post_hashtags(post)
                if any(tag in clean_targets for tag in post_tags):
                    is_match = True
            else:
                # Mode B: Check for keyword/bigram existence in full body text
                post_text = post.get("text", "").lower()
                if post_text and any(kw in post_text for kw in clean_targets):
                    is_match = True

            # 2. Write if a match was found in either mode
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
