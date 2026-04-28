from collections import defaultdict
import orjson
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from config.constants import MAX_WORKERS
from config.paths import PATH_USER_POSTS, GENERAL_PATH
from config.logging import logger
from data_processing.util.functions import get_post_hashtags, iter_jsonl


"""
Creates a dict of all hashtags in the data and their occurrences (ranked in descending order).
"""

OUTPUT_FILE = GENERAL_PATH / "hashtag_counts_new.json"


def find_hashtags_in_file(file_path: Path) -> defaultdict[str, int]:
    local_hashtag_counts = defaultdict(int)

    for data in iter_jsonl(file_path):
        unique_tags = get_post_hashtags(data)

        for tag in unique_tags:
            local_hashtag_counts[tag] += 1

    return local_hashtag_counts


def merge_counts(count_dicts: list[defaultdict[str, int]]) -> defaultdict[str, int]:
    aggregated_counts = defaultdict(int)
    for d in count_dicts:
        for tag, count in d.items():
            aggregated_counts[tag] += count
    return aggregated_counts


def main():
    start_time = time.time()

    if not PATH_USER_POSTS.is_dir():
        logger.error(f"Source directory not found: {PATH_USER_POSTS}")
        return

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    files = list(PATH_USER_POSTS.glob("*.jsonl"))
    total_files = len(files)

    if total_files == 0:
        logger.warning(f"No .jsonl files found in {PATH_USER_POSTS}. Exiting.")
        return

    logger.info(f"Starting aligned hashtag search in {total_files:,} files...")

    all_counts = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(find_hashtags_in_file, f): f for f in files}

        for i, future in enumerate(as_completed(future_to_file)):
            result = future.result()
            if result:
                all_counts.append(result)

            processed_count = i + 1
            if processed_count % 1000 == 0 or processed_count == total_files:
                logger.info(
                    f"Processed {processed_count:,}/{total_files:,} files "
                    f"({(processed_count / total_files) * 100:.2f}%)"
                )

    logger.info("Merging results...")
    final_counts = merge_counts(all_counts)
    sorted_counts = dict(
        sorted(final_counts.items(), key=lambda item: item[1], reverse=True)
    )

    try:
        with open(OUTPUT_FILE, "wb") as out_f:
            out_f.write(orjson.dumps(sorted_counts, option=orjson.OPT_INDENT_2))
    except IOError as e:
        logger.error(f"Could not write to output file {OUTPUT_FILE}: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Done in {elapsed / 60:.2f} minutes. Results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
