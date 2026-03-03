import orjson
import random
from pathlib import Path
from typing import List

from data_processing.config.paths import PATH_USER_POSTS
from data_processing.config.logging import logger

"""
Create a background corpus of English posts that contain no hashtags. The background
corpus is then used to obtain representative keywords for the target political issue.
"""

OUTPUT_PATH = Path("")


def build_background_corpus(
    source_dir: Path, output_path: Path, target_size: int = 500000
):
    """
    Args:
        source_dir: Path object to the folder containing user .jsonl files.
        output_path: Path object where the resulting background corpus will be saved.
        target_size: How many posts to collect (default 200k).
    """

    if not source_dir.exists():
        logger.error(f"Source directory '{source_dir}' does not exist.")
        return

    # 1. Get a list of all user files (assuming .jsonl based on previous scripts)
    logger.info(f"Scanning files in {source_dir}...")
    user_files = list(source_dir.glob("*.jsonl"))

    if not user_files:
        logger.warning(f"No .jsonl files found in {source_dir}.")
        return

    # 2. Shuffle the file list randomly
    # This ensures we don't just get users from 'A' or 'B' partitions.
    random.shuffle(user_files)

    collected_posts: List[str] = []

    logger.info(
        f"Starting collection from {len(user_files):,} files. Target: {target_size:,} posts."
    )

    # 3. Iterate through random user files
    for file_path in user_files:
        if len(collected_posts) >= target_size:
            break

        try:
            with open(file_path, "rb") as f:
                for line in f:
                    if len(collected_posts) >= target_size:
                        break

                    try:
                        post = orjson.loads(line)

                        # --- FILTERING LOGIC ---

                        # 1. Check Language (Strict English only)
                        # We use strict equality to avoid mixed language posts
                        langs = post.get("langs")
                        if isinstance(langs, list) and langs == ["eng"]:
                            # 2. Check Text Content
                            text = post.get("text", "")

                            # 3. No hashtags
                            # We verify the text is not empty and contains no '#' symbol
                            if text and "#" not in text:
                                # Add valid text to corpus
                                collected_posts.append(text)

                    except orjson.JSONDecodeError:
                        continue

        except Exception as e:
            logger.warning(f"Error reading file {file_path.name}: {e}")
            continue

    # 4. Save the result
    logger.info(
        f"Collection complete. Saving {len(collected_posts):,} posts to {output_path}..."
    )

    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f_out:
            # We dump the whole list as a single JSON object (List[str])
            # consistent with how TF-IDF vectorizers typically expect input
            f_out.write(orjson.dumps(collected_posts))

        logger.info("Background corpus successfully saved.")

    except IOError as e:
        logger.error(f"Failed to save output file: {e}")


if __name__ == "__main__":
    # Ensure strict English filtering and random sampling for the background model
    build_background_corpus(PATH_USER_POSTS, OUTPUT_PATH)
