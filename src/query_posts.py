from concurrent.futures import ThreadPoolExecutor, as_completed
import orjson
import logging
import time

from src.config import KEYWORDS, PATH_USER_POSTS, OUTPUT_FILE, MAX_WORKERS, VERBOSE

logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for more details
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/search_log.txt", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def process_file(file_path):
    matches = []
    try:
        with open(file_path, "rb") as f:
            for line in f:
                try:
                    data = orjson.loads(line)
                    text = data.get("text", "").lower()

                    if any(k in text for k in KEYWORDS):
                        matches.append(data)
                        if VERBOSE:
                            logger.info(f"Found a relevant post: {text}")
                except Exception as e:
                    logger.debug(f"JSON parse error in {file_path.name}: {e}")
    except Exception as e:
        logger.error(f"Could not read file {file_path}: {e}")
    return matches


def main():
    start_time = time.time()

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

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
        futures = {executor.submit(process_file, f): f for f in files}
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
    logger.info("Detailed log saved to: search_log.txt")


if __name__ == "__main__":
    main()
