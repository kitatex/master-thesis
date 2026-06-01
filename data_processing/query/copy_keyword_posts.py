from concurrent.futures import ThreadPoolExecutor, as_completed
import orjson
import time
from pathlib import Path

from data_processing.util.functions import (
    extend_with_no_spaces,
    iter_jsonl,
    get_post_hashtags,
    extract_hashtags_from_jsonl,
    extract_keys_from_jsonl,
    optimize_keyword_list,
)
from config.paths import PATH_USER_POSTS, PATH_USER_POSTS_MIN10, TOPICS_PATH
from config.constants import MAX_WORKERS, VERBOSE
from config.logging import logger

"""
Copy the posts which contain specified keywords (typically, hashtags).

KEYWORDS (List[str]): A list of keywords whose occurences will cause a post to be copied from the full dataset.
If len(KEYWORDS) > 1, then the first keyword is used as identifier. Can also be read from a file.

OUTPUT_FILE (.jsonl): Output location.

HASHTAG_ONLY (bool): whether we are looking ONLY at a list of hashtag keywords. If False, a mix of hashtags and non-hashtags also works.
"""


TOPIC = "defundpolice"

HASHTAG_ONLY = False

ADD_NO_WHITESPACE_KEYWORDS = True

# PATH_USER_POSTS (full) or PATH_USER_POSTS_MIN10 (filtered)
POSTS_PATH = PATH_USER_POSTS_MIN10


if HASHTAG_ONLY:
    path = TOPICS_PATH / f"{TOPIC}/co_hashtags/chosen_closest_hashtags.jsonl"
    KEYWORDS = extract_hashtags_from_jsonl(path)
    OUTPUT_FILE = TOPICS_PATH / f"{TOPIC}/co_hashtags/posts.jsonl"
else:
    path = TOPICS_PATH / f"{TOPIC}/keywords/chosen_keywords_bigrams.jsonl"
    path = TOPICS_PATH / f"{TOPIC}/keywords/keywords.jsonl"
    OUTPUT_FILE = TOPICS_PATH / f"{TOPIC}/keywords/posts.jsonl"

    # path = TOPICS_PATH / "_multiple/20260523/keywords.jsonl"
    # OUTPUT_FILE = TOPICS_PATH / "_multiple/20260523/posts.jsonl"

    KEYWORDS = extract_keys_from_jsonl(path)


original_len = len(KEYWORDS)
clean_keywords = optimize_keyword_list([k.lower() for k in KEYWORDS])
logger.info(
    f"Optimized Positive Keywords: {original_len} -> {len(clean_keywords)} (Removed {original_len - len(clean_keywords)} redundant words)"
)
if ADD_NO_WHITESPACE_KEYWORDS:
    clean_keywords = extend_with_no_spaces(clean_keywords)

FORBIDDEN_KEYWORDS_FILE = TOPICS_PATH / f"{TOPIC}/keywords/forbidden_keywords.jsonl"
forbidden_keywords = extract_keys_from_jsonl(FORBIDDEN_KEYWORDS_FILE)
print(forbidden_keywords)


SECONDARY_KEYWORDS_FILE = TOPICS_PATH / f"{TOPIC}/keywords/secondary_keywords.jsonl"
secondary_keywords = []

if SECONDARY_KEYWORDS_FILE.exists():
    secondary_keywords = optimize_keyword_list(
        [k.lower() for k in extract_keys_from_jsonl(SECONDARY_KEYWORDS_FILE)]
    )
    if ADD_NO_WHITESPACE_KEYWORDS:
        secondary_keywords = extend_with_no_spaces(secondary_keywords)
    logger.info(f"Loaded {len(secondary_keywords)} secondary keywords.")
else:
    logger.info("No secondary keywords loaded.")


def find_posts_with_keyword(file_path: Path):
    matches = []

    if HASHTAG_ONLY:
        clean_keywords = {k.lower().lstrip("#") for k in KEYWORDS}
    else:
        clean_keywords = [k.lower() for k in KEYWORDS]

    for data in iter_jsonl(file_path):
        post_text = data.get("text", "").lower()

        # Primary keyword match
        if HASHTAG_ONLY:
            post_tags = get_post_hashtags(data, en_only=False)

            is_match = any(ck in post_tags for ck in clean_keywords)

        else:
            is_match = any(ck in post_text for ck in clean_keywords)

        if not is_match:
            continue

        # Forbidden keyword filter
        if forbidden_keywords:
            if any(fk in post_text for fk in forbidden_keywords):
                continue

        # Secondary keyword requirement
        if secondary_keywords:
            has_secondary_match = any(sk in post_text for sk in secondary_keywords)

            if not has_secondary_match:
                continue

        matches.append(data)

        if VERBOSE:
            logger.info(f"Found a relevant post: {data.get('text')}")

    return matches


def main():
    start_time = time.time()

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    else:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.touch()

    files = list(POSTS_PATH.glob("*.jsonl"))
    total_files = len(files)

    logger.info(
        f"Starting search for {KEYWORDS} in {POSTS_PATH} with {total_files:,} files..."
    )

    logger.info(f"Results will be saved to {OUTPUT_FILE}")

    confirm = input("Proceed? (y/N): ").strip().lower()

    if confirm != "y":
        logger.info("Aborted by user.")
        exit(0)

    processed_count = 0
    match_count = 0

    with (
        ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor,
        open(OUTPUT_FILE, "wb") as out_f,
    ):
        futures = {executor.submit(find_posts_with_keyword, f): f for f in files}

        for future in as_completed(futures):
            results = future.result()

            processed_count += 1

            if results:
                match_count += len(results)

                for item in results:
                    out_f.write(orjson.dumps(item))
                    out_f.write(b"\n")

            if processed_count % 10000 == 0:
                logger.info(
                    f"Processed {processed_count:,}/{total_files:,} files "
                    f"({(processed_count / total_files) * 100:.2f}% done)"
                )

    elapsed = time.time() - start_time

    logger.info(
        f"Search complete in {elapsed / 60:.2f} min — {match_count:,} matching entries found."
    )

    logger.info(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
