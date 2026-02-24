import csv
import time
import orjson
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from query.config.paths import PATH_USER_POSTS, TOPICS_OUTPUT_PATH
from query.config.constants import MAX_WORKERS, VERBOSE
from query.config.logging import logger

# --- File Paths ---
HASHTAGS_CSV_PATH = Path("C:/Users/leond/Desktop/univie_sie_input/hashtags_raw.csv")

TOPICS_CSV_PATH = Path("C:/Users/leond/Desktop/univie_sie_input/topic_mappings.csv")

OUTPUT_FILE = TOPICS_OUTPUT_PATH / "univie/topic_posts.jsonl"


def load_mappings() -> tuple[dict[str, int], dict[int, str]]:
    """Loads the CSV files and returns the necessary mapping dictionaries."""
    topic_idx_to_name = {}
    with open(TOPICS_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) == 2:
                topic_name, idx = row[0].strip(), int(row[1].strip())
                topic_idx_to_name[idx] = topic_name

    keyword_to_topic_idx = {}
    with open(HASHTAGS_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) == 2:
                keyword = f"#{row[0].strip().lower()}"
                idx = int(row[1].strip())
                keyword_to_topic_idx[keyword] = idx

    return keyword_to_topic_idx, topic_idx_to_name


# Load mappings globally so thread workers can access them
KEYWORD_TO_TOPIC_IDX, TOPIC_IDX_TO_NAME = load_mappings()


def find_posts_with_topics(file_path: Path):
    matches = []
    try:
        with open(file_path, "rb") as f:
            for line in f:
                try:
                    data = orjson.loads(line)
                    text = data.get("text", "").lower()

                    matched_topic_indices = set()

                    # Check for all keywords in the text
                    for keyword, topic_idx in KEYWORD_TO_TOPIC_IDX.items():
                        if keyword in text:
                            matched_topic_indices.add(topic_idx)

                    # If we found at least one matching keyword
                    if matched_topic_indices:
                        # Determine the topic string
                        if len(matched_topic_indices) == 1:
                            assigned_idx = matched_topic_indices.pop()
                            data["topic"] = TOPIC_IDX_TO_NAME.get(
                                assigned_idx, "unknown"
                            )
                        else:
                            data["topic"] = "multiple"

                        matches.append(data)

                        if VERBOSE:
                            logger.info(
                                f"Found a relevant post (Topic: {data['topic']}): {text}"
                            )

                except Exception as e:
                    logger.debug(f"JSON parse error in {file_path.name}: {e}")
    except Exception as e:
        logger.error(f"Could not read file {file_path}: {e}")

    return matches


def main():
    start_time = time.time()

    # Prepare output directory and file
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    else:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.touch()

    files = list(PATH_USER_POSTS.glob("*.jsonl"))
    total_files = len(files)

    logger.info(
        f"Loaded {len(KEYWORD_TO_TOPIC_IDX)} keywords and {len(TOPIC_IDX_TO_NAME)} topics."
    )
    logger.info(f"Starting search in {PATH_USER_POSTS} across {total_files:,} files...")
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
        futures = {executor.submit(find_posts_with_topics, f): f for f in files}
        for future in as_completed(futures):
            results = future.result()
            processed_count += 1
            if results:
                match_count += len(results)
                for item in results:
                    out_f.write(orjson.dumps(item))
                    out_f.write(b"/n")

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
