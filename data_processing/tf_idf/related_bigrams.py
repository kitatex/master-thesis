import orjson
import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from data_processing.util.functions import extract_keys_from_jsonl
from config.paths import BACKGROUND_CORPUS_PATH, TOPICS_PATH
from config.logging import logger

"""
Ordered Bigram Generator: Captures bigrams in their original sequence.
Filters out standard English stopwords and previously identified unigrams.
"""

TOPIC_NAME = "#gamedev"

SEED_HASHTAG_CORPUS = (
    TOPICS_PATH / f"{TOPIC_NAME}/hashtag_corpus/posts_merged_deduplicated.jsonl"
)
OUTPUT_JSONL = TOPICS_PATH / f"{TOPIC_NAME}/keywords/ranked_bigrams.jsonl"

TOP_N_KEYWORDS = 50
SEEDS_TO_EXCLUDE = None

# --- STOPWORD LOGIC ---
unigram_path = TOPICS_PATH / f"{TOPIC_NAME}/keywords/chosen_keywords.jsonl"
previous_unigrams = set(extract_keys_from_jsonl(unigram_path))
ALL_STOPWORDS = ENGLISH_STOP_WORDS.union(previous_unigrams)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"(https?://|www\.)\S+", "", text)
    text = re.sub(r"\b\S+\.(com|app|net|org|profile|html|php)\b", "", text)
    text = re.sub(r"\d+", "", text)
    return text


def bigram_analyzer(doc):
    """
    Custom analyzer that:
    1. Cleans the text.
    2. Tokenizes.
    3. Filters out the combined stopword list.
    4. Generates bigrams in ORIGINAL document order.
    """
    cleaned = clean_text(doc)
    # Tokenize words with 2+ characters
    tokens = re.findall(r"\b\w\w+\b", cleaned)

    # Filter stopwords (Standard + your previous unigrams)
    tokens = [t for t in tokens if t not in ALL_STOPWORDS]

    bigrams = []
    for i in range(len(tokens) - 1):
        # We take the sequence as it appears in the text
        gram = " ".join(tokens[i : i + 2])
        bigrams.append(gram)
    return bigrams


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
    top_n: int = 100,
    min_df: int = 15,
    max_df: float = 0.3,
    exclude_words: Optional[List[str]] = None,
) -> pd.DataFrame:
    logger.info("Loading corpora...")
    topic_posts = load_corpus(topic_corpus_path)
    background_posts = load_corpus(background_corpus_path)

    if not topic_posts or not background_posts:
        return pd.DataFrame()

    all_posts = topic_posts + background_posts
    y = np.array([1] * len(topic_posts) + [0] * len(background_posts))

    logger.info("Vectorizing with ordered bigram analyzer...")

    vectorizer = TfidfVectorizer(
        analyzer=bigram_analyzer,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(all_posts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    logger.info(f"Generated {len(feature_names):,} unique ordered bigram features.")
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

        # Check if any parts of the bigram are in the manual exclude set
        if not any(excluded in word.split() for excluded in exclude_set):  # type: ignore
            top_keywords.append(word)
            top_scores.append(score)
            count += 1
        if count >= top_n:
            break

    return pd.DataFrame({"keyword": top_keywords, "score": top_scores})


if __name__ == "__main__":
    df = get_representative_keywords(
        SEED_HASHTAG_CORPUS,
        BACKGROUND_CORPUS_PATH,
        top_n=TOP_N_KEYWORDS,
        exclude_words=SEEDS_TO_EXCLUDE,
    )

    if not df.empty:
        # Enforce lowercase keys for the JSON output
        output_dict = {
            str(k).lower(): float(v) for k, v in zip(df["keyword"], df["score"])
        }

        OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_JSONL, "wb") as f:
            f.write(orjson.dumps(output_dict, option=orjson.OPT_INDENT_2))
        logger.info(f"Ordered bigram list saved to {OUTPUT_JSONL}")
