import time
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Assuming these are available in your environment based on the previous scripts
from config.paths import PATH_USER_POSTS
from config.constants import MAX_WORKERS
from config.logging import logger

"""
Filters user .jsonl files based on a minimum post threshold.
Copies files that meet the threshold to a new directory.
"""

# --- CONFIGURATION ---
MIN_POSTS = 10

output_path = (
    "C:/Users/leond/Documents/03-FILES/01 data/Bluesky Data filtered/user_posts"
)
OUTPUT_DIR = Path(output_path)


def process_user_file(file_path: Path, output_dir: Path, min_posts: int) -> bool:
    """
    Reads the file line-by-line. If it hits the min_posts threshold,
    it immediately copies the file and returns True (early exit).
    """
    try:
        valid_post_count = 0
        with open(file_path, "rb") as f:
            for line in f:
                # Basic check to ignore empty lines
                if line.strip():
                    valid_post_count += 1

                    # Early exit: we hit the threshold, no need to read the rest
                    if valid_post_count >= min_posts:
                        # Copy the file to the new location, preserving metadata
                        target_path = output_dir / file_path.name
                        shutil.copy2(file_path, target_path)
                        return True

        # If the loop finishes naturally, they didn't meet the threshold
        return False

    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return False


def main():
    start_time = time.time()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(PATH_USER_POSTS.glob("*.jsonl"))
    total_files = len(files)

    logger.info(
        f"Checking {total_files:,} users for minimum activity (k={MIN_POSTS})..."
    )
    logger.info(f"Eligible files will be copied to: {OUTPUT_DIR}")

    confirm = input("Proceed? (y/N): ").strip().lower()
    if confirm != "y":
        logger.info("Aborted by user.")
        return

    processed_count = 0
    copied_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_user_file, f, OUTPUT_DIR, MIN_POSTS): f
            for f in files
        }

        # Process results as they complete
        for future in as_completed(futures):
            processed_count += 1

            # process_user_file returns True if the file was copied
            if future.result():
                copied_count += 1

            # Periodic logging
            if processed_count % 10000 == 0:
                logger.info(
                    f"Processed {processed_count:,}/{total_files:,} users "
                    f"({(processed_count / total_files) * 100:.2f}% done). "
                    f"Copied so far: {copied_count:,}"
                )

    elapsed = time.time() - start_time

    logger.info("--- Filter Complete ---")
    logger.info(f"Total users processed: {total_files:,}")
    logger.info(f"Users retained (>= {MIN_POSTS} posts): {copied_count:,}")
    logger.info(f"Users dropped: {total_files - copied_count:,}")
    logger.info(f"Time elapsed: {elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
