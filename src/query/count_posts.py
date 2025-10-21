import time
import orjson
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from builtins import enumerate

from src.config.constants import MAX_WORKERS
from src.config.paths import PATH_USER_POSTS, GENERAL_OUTPUT_PATH
from src.config.logging import logger


OUTPUT_FILE = GENERAL_OUTPUT_PATH / "post_counts.json"


def count_posts_in_file(file_path: Path) -> int:
    """Counts the number of lines (posts) in a .jsonl file."""
    try:
        with file_path.open("rb") as f:
            # This is a fast method for counting lines in a file
            return sum(1 for _ in f)
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return 0


def main():
    start_time = time.time()

    # Ensure the source directory exists
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

    logger.info(f"Starting post count in {total_files:,} files...")
    logger.info(f"Results will be saved to {OUTPUT_FILE}")

    # --- Processing ---
    total_post_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all files to the executor with the new counting function
        future_to_file = {executor.submit(count_posts_in_file, f): f for f in files}

        for i, future in enumerate(as_completed(future_to_file)):
            # Add the result from each completed future to the total count
            post_count = future.result()
            total_post_count += post_count

            # Log progress periodically
            processed_count = i + 1
            if processed_count % 1000 == 0 or processed_count == total_files:
                logger.info(
                    f"Processed {processed_count:,}/{total_files:,} files "
                    f"({(processed_count / total_files) * 100:.2f}%)"
                )

    # --- Saving ---
    logger.info("Processing complete. Saving results...")

    # Create the final dictionary for the JSON output
    final_result = {"number of posts": total_post_count}

    try:
        with open(OUTPUT_FILE, "wb") as out_f:
            # Use orjson for fast JSON serialization
            out_f.write(orjson.dumps(final_result, option=orjson.OPT_INDENT_2))
    except IOError as e:
        logger.error(f"Could not write to output file {OUTPUT_FILE}: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"Count complete in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)."
    )
    logger.info(f"Found a total of {total_post_count:,} posts.")
    logger.info(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
