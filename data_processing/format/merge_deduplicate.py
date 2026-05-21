import tempfile
import os
from pathlib import Path
from typing import Union

from data_processing.format.deduplicate import deduplicate_posts_by_id
from config.logging import logger
from config.paths import TOPICS_PATH


TOPIC_NAME = "#climatechange"

# True, if merging seed and co-hashtags
# False, if merging hashtag corpus and keyword posts
HASHTAG_ONLY = False


if HASHTAG_ONLY:
    # --- HASHTAGS ---
    FILE_1 = TOPICS_PATH / f"{TOPIC_NAME}/seed_hashtag/posts.jsonl"
    FILE_2 = TOPICS_PATH / f"{TOPIC_NAME}/co_hashtags/posts.jsonl"
    OUTPUT_FILE = (
        TOPICS_PATH / f"{TOPIC_NAME}/hashtag_corpus/posts_merged_deduplicated.jsonl"
    )

else:
    # --- KEYWORDS ---
    FILE_1 = (
        TOPICS_PATH / f"{TOPIC_NAME}/hashtag_corpus/posts_merged_deduplicated.jsonl"
    )
    FILE_2 = TOPICS_PATH / f"{TOPIC_NAME}/keywords/posts.jsonl"
    OUTPUT_FILE = TOPICS_PATH / f"{TOPIC_NAME}/full/posts_merged_deduplicated.jsonl"


def merge_and_deduplicate_jsonl(
    file1_path: Union[str, Path],
    file2_path: Union[str, Path],
    output_path: Union[str, Path],
    id_key: str = "post_id",
):
    """
    Merges two .jsonl files and then deduplicates the resulting posts.
    Uses a temporary file to ensure we don't load massive datasets into RAM.
    """
    file1_path = Path(file1_path)
    file2_path = Path(file2_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not file1_path.exists() or not file2_path.exists():
        logger.error(
            f"Error: One or both input files missing.\n1: {file1_path}\n2: {file2_path}"
        )
        return

    logger.info(f"Merging '{file1_path.name}' and '{file2_path.name}'...")

    # Create a temporary file to hold the raw merged data
    temp_fd, temp_path_str = tempfile.mkstemp(suffix=".jsonl")
    temp_path = Path(temp_path_str)

    try:
        with open(temp_fd, "wb") as temp_file:
            # Stream file 1
            with open(file1_path, "rb") as f1:
                for line in f1:
                    if line.strip():
                        temp_file.write(line)

            # Stream file 2
            with open(file2_path, "rb") as f2:
                for line in f2:
                    if line.strip():
                        temp_file.write(line)

        logger.info("Merge complete. Beginning deduplication...")

        # Pass the merged temporary file to your existing deduplication function
        deduplicate_posts_by_id(temp_path, output_path, id_key=id_key)

        logger.info("Merge and deduplication pipeline finished successfully.")

    finally:
        # Always clean up the temporary file, even if the script crashes
        if temp_path.exists():
            os.remove(temp_path)


if __name__ == "__main__":
    merge_and_deduplicate_jsonl(FILE_1, FILE_2, OUTPUT_FILE)
