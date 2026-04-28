from pathlib import Path
import orjson
import math
from collections import defaultdict

from data_processing.util.functions import find_hashtags
from config.logging import logger
from config.constants import TOTAL_POSTS_NUMBER
from config.paths import HT_COUNTS_PATH, TOPICS_PATH

"""
After obtaining a corpus for a seed hashtag, this script can be used to related
co-hashtags. The HT_COUNTS_PATH file is used to normalize using tf-idf.

INPUT_PATH (.jsonl): Path to the seed hashtag corpus.

OUTPUT_PATH (.jsonl): Path where the ranked hashtag list will be saved.
"""


TOPIC_NAME = "#gaza"  # e.g., #climatecrisis

INPUT_PATH = TOPICS_PATH / f"{TOPIC_NAME}/seed_hashtag/posts_deduplicated.jsonl"

OUTPUT_PATH = TOPICS_PATH / f"{TOPIC_NAME}/co_hashtags/closest_hashtags.jsonl"


def compute_tf(
    posts_path: Path,
):
    if not posts_path.exists():
        logger.error(f"Error: Input file not found at '{posts_path}'")
        return

    tf_counts = defaultdict(int)

    with open(posts_path, "rb") as f:
        for line in f:
            data = orjson.loads(line)

            text = data.get("text", "")
            if isinstance(text, str):
                hashtags_found = find_hashtags(text)
                for tag in hashtags_found:
                    tf_counts[tag.lower()] += 1

    return tf_counts


def obtain_closest_hashtags(tf_counts: dict, ht_counts_path: Path):
    if not ht_counts_path.exists():
        logger.error(f"Error: Input file not found at '{ht_counts_path}'")
        return

    with open(ht_counts_path, "rb") as f:
        ht_counts_dict = dict(orjson.loads(f.read()))

    tfidf_scores = {}
    for ht, term_frequency in tf_counts.items():
        doc_frequency = ht_counts_dict.get(ht, 1)

        # Avoid division by zero and log of zero issues.
        if doc_frequency > 0:
            idf = math.log(TOTAL_POSTS_NUMBER / doc_frequency)
            tfidf_scores[ht] = term_frequency * idf
        else:
            print(doc_frequency)
            raise ValueError(
                f"The doc_frequency of {ht} is 0, which should never occur since we are using subsets of the main corpus."
            )

    # Sort by TF-IDF score, descending
    sorted_hashtags = dict(
        sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_hashtags


def main():
    tf_results = compute_tf(INPUT_PATH)
    if tf_results:
        closest_hashtags = obtain_closest_hashtags(tf_results, HT_COUNTS_PATH)

    if closest_hashtags:
        try:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

            with open(OUTPUT_PATH, "wb") as out_f:
                out_f.write(orjson.dumps(closest_hashtags, option=orjson.OPT_INDENT_2))
            logger.info(f"Results saved to: {OUTPUT_PATH}")
            logger.info("\n--- Output ---")
            logger.info(
                orjson.dumps(closest_hashtags, option=orjson.OPT_INDENT_2).decode()
            )
        except IOError as e:
            logger.error(f"Could not write to output file {OUTPUT_PATH}: {e}")


if __name__ == "__main__":
    main()
