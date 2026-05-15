from concurrent.futures import ThreadPoolExecutor, as_completed
import orjson
import time
from pathlib import Path

from data_processing.util.functions import (
    iter_jsonl,
    get_post_hashtags,
    extract_hashtags_from_jsonl,
    extract_keys_from_jsonl,
)
from config.paths import PATH_USER_POSTS, TOPICS_PATH
from config.constants import MAX_WORKERS, VERBOSE
from config.logging import logger

"""
Copy the posts which contain specified keywords (typically, hashtags).

KEYWORDS (List[str]): A list of keywords whose occurences will cause a post to be copied from the full dataset.
If len(KEYWORDS) > 1, then the first keyword is used as identifier. Can also be read from a file.

OUTPUT_FILE (.jsonl): Output location.
"""


# KEYWORDS = ["#gaza", "#climatechange"]

HASHTAG_ONLY = False

if HASHTAG_ONLY:
    TOPIC = "#climatechange"
    path = TOPICS_PATH / f"{TOPIC}/co_hashtags/chosen_closest_hashtags.jsonl"
    KEYWORDS = extract_hashtags_from_jsonl(path)
    OUTPUT_FILE = TOPICS_PATH / f"{TOPIC}/co_hashtags/posts.jsonl"
else:
    path = TOPICS_PATH / "#climatechange/keywords/chosen_keywords_bigrams.jsonl"
    KEYWORDS = extract_keys_from_jsonl(path)
    OUTPUT_FILE = TOPICS_PATH / "#climatechange/keywords/posts.jsonl"


def find_posts_with_keyword(file_path: Path):
    matches = []

    if HASHTAG_ONLY:
        # Prepare keywords: lowercase and remove leading '#'
        clean_keywords = {k.lower().lstrip("#") for k in KEYWORDS}

    else:
        # Prepare keywords for plain string matching
        clean_keywords = [k.lower() for k in KEYWORDS]

    # Use the shared generator to handle file IO and error logging
    for data in iter_jsonl(file_path):
        if HASHTAG_ONLY:
            # Extract hashtags using the shared regex/language logic
            post_tags = get_post_hashtags(data)

            # Check hashtag intersection
            is_match = any(ck in post_tags for ck in clean_keywords)

        else:
            # Plain text matching
            post_text = data.get("text", "").lower()

            is_match = any(ck in post_text for ck in clean_keywords)

        if is_match:
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

    files = list(PATH_USER_POSTS.glob("*.jsonl"))
    total_files = len(files)

    logger.info(
        f"Starting search for {KEYWORDS} in {PATH_USER_POSTS} with {total_files:,} files..."
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
