"""Creates a dict of all hashtags in the data and their occurences"""

from collections import defaultdict
import orjson
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from query.config.constants import MAX_WORKERS
from query.config.paths import PATH_USER_POSTS, GENERAL_OUTPUT_PATH
from query.config.logging import logger
from query.util.functions import find_hashtags

# --- Manual ---
# Adjust these paths and settings as needed
OUTPUT_FILE = GENERAL_OUTPUT_PATH / "hashtag_counts.json"


def find_hashtags_in_file(file_path: Path) -> defaultdict[str, int]:
    local_hashtag_counts = defaultdict(int)
    try:
        with open(file_path, "rb") as f:
            for line in f:
                data = orjson.loads(line)
                # Ensure 'text' exists and is a string
                text = data.get("text", "")
                if isinstance(text, str):
                    # Find all matching hashtags in the text
                    hashtags_found = find_hashtags(text)
                    for tag in hashtags_found:
                        # Aggregate counts in lowercase
                        local_hashtag_counts[tag.lower()] += 1

    except IOError as e:
        logger.error(f"Could not read file {file_path}: {e}")
    return local_hashtag_counts


def merge_counts(count_dicts: list[defaultdict[str, int]]) -> defaultdict[str, int]:
    """Merges a list of count dictionaries into a single one."""
    aggregated_counts = defaultdict(int)
    for d in count_dicts:
        for tag, count in d.items():
            aggregated_counts[tag] += count
    return aggregated_counts


def main():
    start_time = time.time()

    # Ensure the target directory exists
    if not PATH_USER_POSTS.is_dir():
        logger.error(f"Source directory not found: {PATH_USER_POSTS}")
        return

    # Delete old output file if it exists
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    files = list(PATH_USER_POSTS.glob("*.jsonl"))
    total_files = len(files)

    if total_files == 0:
        logger.warning(f"No .jsonl files found in {PATH_USER_POSTS}. Exiting.")
        return

    logger.info(f"Starting hashtag search in {total_files:,} files...")
    logger.info(f"Results will be saved to {OUTPUT_FILE}")

    # --- Processing ---
    all_counts = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all files to the executor
        future_to_file = {executor.submit(find_hashtags_in_file, f): f for f in files}

        for i, future in enumerate(as_completed(future_to_file)):
            # Append the result from each completed future
            result = future.result()
            if result:
                all_counts.append(result)

            # Log progress periodically
            processed_count = i + 1
            if processed_count % 1000 == 0 or processed_count == total_files:
                logger.info(
                    f"Processed {processed_count:,}/{total_files:,} files "
                    f"({(processed_count / total_files) * 100:.2f}%)"
                )

    # --- Aggregation and Saving ---
    logger.info("Merging results from all files...")
    final_counts = merge_counts(all_counts)

    # Sort by count, descending
    sorted_counts = dict(
        sorted(final_counts.items(), key=lambda item: item[1], reverse=True)
    )

    try:
        with open(OUTPUT_FILE, "wb") as out_f:
            # Use orjson for fast JSON serialization
            out_f.write(orjson.dumps(sorted_counts, option=orjson.OPT_INDENT_2))
    except IOError as e:
        logger.error(f"Could not write to output file {OUTPUT_FILE}: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"Search complete in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)."
    )
    logger.info(f"Found {len(final_counts):,} unique hashtags.")
    logger.info(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
