import orjson
import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

from config.paths import BACKGROUND_CORPUS_PATH, TOPICS_PATH
from config.logging import logger

"""
Based on a hashtag corpus file (seed + co-hashtags), obtain the top n related keywords.
This version strips all #hashtags from the text and ensures all output keys 
are lowercased for easier downstream matching.
"""

TOPIC_NAME = "#climatechange"  # identified by seed hashtag; e.g., #gaza

HASHTAG_CORPUS = (
    TOPICS_PATH / f"{TOPIC_NAME}/hashtag_corpus/posts_merged_deduplicated.jsonl"
)

OUTPUT_JSONL = TOPICS_PATH / f"{TOPIC_NAME}/keywords/ranked_keywords.jsonl"

TOP_N_KEYWORDS = 100
SEEDS_TO_EXCLUDE = None


def remove_hashtags(text: str) -> str:
    """
    Removes any word starting with # (e.g., '#standwithukraine' -> '')
    to ensure only body text is analyzed.
    """
    if not isinstance(text, str):
        return ""
    # This regex removes the # and all alphanumeric/underscore characters following it
    return re.sub(r"#\w+", "", text)


def load_corpus(file_path: Path) -> List[str]:
    corpus = []
    try:
        with open(file_path, "rb") as f:
            content = f.read().strip()
            if content.startswith(b"[") and content.endswith(b"]"):
                data = orjson.loads(content)
                if isinstance(data, list):
                    corpus = [str(text) for text in data if text]
            else:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        post_data = orjson.loads(line)
                        text = (
                            post_data.get("text", "")
                            if isinstance(post_data, dict)
                            else str(post_data)
                        )
                        if text:
                            corpus.append(text)
                    except orjson.JSONDecodeError:
                        continue
        return corpus
    except Exception as e:
        logger.error(f"Failed to load corpus from {file_path}: {e}")
        return []


def get_representative_keywords(
    topic_corpus_path: Path,
    background_corpus_path: Path,
    top_n: int = 50,
    min_df: int = 10,
    max_df: float = 0.5,
    exclude_words: Optional[List[str]] = None,
) -> pd.DataFrame:
    logger.info("Loading corpora...")
    topic_posts = load_corpus(topic_corpus_path)
    background_posts = load_corpus(background_corpus_path)

    if not topic_posts or not background_posts:
        return pd.DataFrame()

    all_posts = topic_posts + background_posts
    y = np.array([1] * len(topic_posts) + [0] * len(background_posts))

    logger.info("Vectorizing text (stripping hashtags)...")

    # lowercase=True is the default, but explicitly kept here for clarity.
    vectorizer = TfidfVectorizer(
        preprocessor=remove_hashtags,
        stop_words="english",
        lowercase=True,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(all_posts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    logger.info("Calculating contrastive scores...")
    mean_topic = np.asarray(X[y == 1].mean(axis=0)).flatten()  # type: ignore
    mean_background = np.asarray(X[y == 0].mean(axis=0)).flatten()  # type: ignore
    scores = mean_topic - mean_background

    sorted_indices = scores.argsort()[::-1]
    top_keywords, top_scores = [], []
    exclude_set = set(w.lower() for w in (exclude_words or []))

    count = 0
    for idx in sorted_indices:
        word = feature_names[idx]
        score = scores[idx]
        if score <= 0:
            break
        if word not in exclude_set:
            top_keywords.append(word)
            top_scores.append(score)
            count += 1
        if count >= top_n:
            break

    return pd.DataFrame({"keyword": top_keywords, "score": top_scores})


if __name__ == "__main__":
    df = get_representative_keywords(
        HASHTAG_CORPUS,
        BACKGROUND_CORPUS_PATH,
        top_n=TOP_N_KEYWORDS,
        exclude_words=SEEDS_TO_EXCLUDE,
    )

    if not df.empty:
        # Enforce lowercase on all keys during dictionary creation
        output_dict = {
            str(k).lower(): float(v) for k, v in zip(df["keyword"], df["score"])
        }

        OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSONL, "wb") as f:
            f.write(orjson.dumps(output_dict, option=orjson.OPT_INDENT_2))
        logger.info(f"Cleaned lowercase keyword dictionary saved to {OUTPUT_JSONL}")
