import orjson
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer

# Adjust these imports to match your project structure
from src.config.paths import (
    BACKGROUND_CORPUS_PATH,
    CURRENT_INPUT_PATH,
    CURRENT_OUTPUT_PATH,
)
from src.config.logging import logger


# --- Set Paths ---

# The hashtag corpus
TOPIC_FILE = CURRENT_INPUT_PATH

# Where to save the resulting csv
OUTPUT_CSV = CURRENT_OUTPUT_PATH

# Optional: exclude the seed hashtags themselves (exclusively finds new words)
TOP_N_KEYWORDS = 100
SEEDS_TO_EXCLUDE = None  # e.g., ["climatechange", "climate"]


def load_corpus(file_path: Path) -> List[str]:
    """
    Robust loader that handles both a single JSON list or JSONL (line-by-line).
    """
    corpus = []
    try:
        with open(file_path, "rb") as f:
            content = f.read().strip()

            # Case 1: The file is a single JSON list (Background Corpus)
            if content.startswith(b"[") and content.endswith(b"]"):
                data = orjson.loads(content)
                if isinstance(data, list):
                    # Filter for non-empty strings
                    corpus = [str(text) for text in data if text]
                    logger.info(f"Loaded {len(corpus):,} posts from JSON list format.")

            # Case 2: The file is JSON Lines (Topic Posts / hashtag_corpus)
            else:
                f.seek(0)  # Reset to beginning of file
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        post_data = orjson.loads(line)
                        # Extract 'text' if it's a dict, otherwise take raw string
                        text = (
                            post_data.get("text", "")
                            if isinstance(post_data, dict)
                            else str(post_data)
                        )
                        if text:
                            corpus.append(text)
                    except orjson.JSONDecodeError:
                        continue
                logger.info(f"Loaded {len(corpus):,} posts from JSONL format.")

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
    """
    Identifies keywords that are highly representative of the topic corpus
    relative to the background corpus using a contrastive TF-IDF score.

    Args:
        topic_corpus_path: Path to the JSON list of topic-specific posts.
        background_corpus_path: Path to the JSON list of background posts.
        top_n: Number of top keywords to return.
        min_df: Ignore terms that appear in fewer than 'min_df' documents.
        max_df: Ignore terms that appear in more than 'max_df' (percentage) of documents.
        exclude_words: List of words to explicitly remove from results (e.g., seed hashtags).

    Returns:
        DataFrame containing the top keywords and their scores.
    """

    # 1. Load Data
    logger.info("Loading corpora...")
    topic_posts = load_corpus(topic_corpus_path)
    background_posts = load_corpus(background_corpus_path)

    if not topic_posts or not background_posts:
        logger.error("One or both corpora are empty. Exiting.")
        return pd.DataFrame()

    logger.info(
        f"Loaded {len(topic_posts):,} topic posts and {len(background_posts):,} background posts."
    )

    # 2. Prepare Labels and Combined Corpus
    # We map Topic = 1, Background = 0
    all_posts = topic_posts + background_posts
    y = np.array([1] * len(topic_posts) + [0] * len(background_posts))

    # 3. Vectorization (TF-IDF)
    # We use a custom token pattern to keep words but exclude special characters/numbers if needed.
    # The default pattern usually works well for English.
    logger.info("Vectorizing text (this may take a moment)...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,  # Applies log scaling to term frequency (1 + log(tf)) - good for tweets
    )

    X = vectorizer.fit_transform(all_posts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # 4. Calculate Contrastive Scores
    logger.info("Calculating contrastive scores...")

    # Calculate the mean TF-IDF score for the Topic class (rows where y==1)
    # Note: We convert sparse matrix mean to a dense array/matrix
    mean_topic = np.asarray(X[y == 1].mean(axis=0)).flatten()  # type: ignore

    # Calculate the mean TF-IDF score for the Background class (rows where y==0)
    mean_background = np.asarray(X[y == 0].mean(axis=0)).flatten()  # type: ignore

    # The "Score" is the difference.
    # High positive value = frequent in Topic, rare/absent in Background.
    scores = mean_topic - mean_background

    # 5. Ranking and Filtering
    # Sort indices by score descending
    sorted_indices = scores.argsort()[::-1]

    top_keywords = []
    top_scores = []

    # Normalize exclude_words to lowercase for comparison
    exclude_set = set(w.lower() for w in (exclude_words or []))

    count = 0
    for idx in sorted_indices:
        word = feature_names[idx]
        score = scores[idx]

        # Stop if score becomes non-positive (means it's more common in background)
        if score <= 0:
            break

        if word not in exclude_set:
            top_keywords.append(word)
            top_scores.append(score)
            count += 1

        if count >= top_n:
            break

    # 6. Create Result DataFrame
    results_df = pd.DataFrame({"keyword": top_keywords, "score": top_scores})

    return results_df


if __name__ == "__main__":
    df = get_representative_keywords(
        TOPIC_FILE,
        BACKGROUND_CORPUS_PATH,
        top_n=TOP_N_KEYWORDS,
        exclude_words=SEEDS_TO_EXCLUDE,
    )

    if not df.empty:
        print("\nTop 20 Representative Keywords:")
        print(df.head(20))

        # Save to CSV for manual inspection (Step 7 in your readme)
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Full ranked list saved to {OUTPUT_CSV}")
