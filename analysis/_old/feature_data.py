import pandas as pd
import numpy as np
import json
import ast
import re
from tqdm import tqdm

from config.paths import TOPICS_PATH, MBFC_CSV_PATH, PATH_USER_POSTS

# config
TOPIC_NAME = "#gamedev"

# input
INPUT_URLS_CSV = TOPICS_PATH / f"{TOPIC_NAME}/metadata/extracted_urls_minrp2.csv"
POSTS_DISCUSSION = TOPICS_PATH / f"{TOPIC_NAME}/full/posts_merged_deduplicated.jsonl"

# output
OUTPUT_GLOBAL_CSV = (
    TOPICS_PATH / f"{TOPIC_NAME}/metadata/features/{TOPIC_NAME}_features_global.csv"
)
OUTPUT_DISC_CSV = (
    TOPICS_PATH / f"{TOPIC_NAME}/metadata/features/{TOPIC_NAME}_features_discussion.csv"
)

BIAS_MAP = {
    "left": -1.5,
    "left-center": -0.5,
    "neutral": 0.0,
    "right-center": 0.5,
    "right": 1.5,
}

CURSE_WORDS = ["fuck", "shit", "bitch", "asshole", "dick", "crap", "bastard"]


def safe_parse_list(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        try:
            return json.loads(val.replace('""', '"'))
        except (json.JSONDecodeError, TypeError):
            return []


def analyze_text_and_meta(line, curse_words_list):
    """Parses a raw post line and extracts text-level behavior flags."""
    try:
        post = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None

    text = post.get("text", "")
    text_lower = text.lower()

    # Check text features
    has_hashtag = 1 if "#" in text else 0
    has_curse = 1 if any(cw in text_lower for cw in curse_words_list) else 0
    post_length = len(text)

    # Check interaction type (Bluesky schema typically includes 'reply' fields)
    is_reply = 1 if post.get("reply") or "reply" in post else 0

    # Check if post text contains an explicit link string
    url_pattern = r"(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"
    has_url = 1 if re.search(url_pattern, text) else 0

    return {
        "has_hashtag": has_hashtag,
        "has_curse": has_curse,
        "post_length": post_length,
        "is_reply": is_reply,
        "has_url": has_url,
    }


def compile_profile_metrics(metrics_list):
    """Aggregates raw post flags into standardized user profile percentages."""
    if not metrics_list:
        return {
            "hashtag_post_ratio": 0.0,
            "pct_url": 0.0,
            "pct_curse": 0.0,
            "reply_post_ratio": 0.0,
            "avg_post_length": 0.0,
            "total_posts": 0,
        }
    df = pd.DataFrame(metrics_list)
    n_posts = len(df)
    return {
        "hashtag_post_ratio": df["has_hashtag"].sum() / n_posts,
        "pct_url": (df["has_url"].sum() / n_posts) * 100,
        "pct_curse": (df["has_curse"].sum() / n_posts) * 100,
        "reply_post_ratio": (df["is_reply"].sum() / n_posts) * 100,
        "avg_post_length": df["post_length"].mean(),
        "total_posts": n_posts,
    }


def calculate_mbfc_metrics(url_list, mbfc_dict):
    if not url_list:
        return {
            "pct_not_high_factual": 0.0,
            "pct_not_high_credibility": 0.0,
            "pct_not_high_popularity": 0.0,
            "bias_mean": np.nan,
            "bias_variance": np.nan,
            "bias_distribution": [],  # Explicit empty fallback
        }

    n_total = len(url_list)
    not_high_fact = 0
    not_high_cred = 0
    not_high_pop = 0
    bias_scores = []

    for domain in url_list:
        meta = mbfc_dict.get(domain, {})
        if "high" not in str(meta.get("factual_reporting", "")).lower():
            not_high_fact += 1
        if "high" not in str(meta.get("mbfc_credibility_rating", "")).lower():
            not_high_cred += 1
        if "high" not in str(meta.get("popularity", "")).lower():
            not_high_pop += 1

        bias_label = str(meta.get("bias", "")).lower()
        if bias_label in BIAS_MAP:
            bias_scores.append(BIAS_MAP[bias_label])

    return {
        "pct_not_high_factual": (not_high_fact / n_total) * 100,
        "pct_not_high_credibility": (not_high_cred / n_total) * 100,
        "pct_not_high_popularity": (not_high_pop / n_total) * 100,
        "bias_mean": np.mean(bias_scores) if bias_scores else np.nan,
        "bias_variance": np.var(bias_scores)
        if len(bias_scores) > 1
        else 0.0
        if len(bias_scores) == 1
        else np.nan,
        "bias_distribution": bias_scores,  # Append the raw mapped distribution
    }


def main():
    mbfc_dict = pd.read_csv(MBFC_CSV_PATH).set_index("source").to_dict(orient="index")
    urls_df = pd.read_csv(INPUT_URLS_CSV)
    urls_df["user_id"] = urls_df["user_id"].astype(str).str.strip()
    urls_df["global_urls"] = urls_df["global_urls"].apply(safe_parse_list)
    urls_df["discussion_urls"] = urls_df["discussion_urls"].apply(safe_parse_list)
    relevant_users = set(urls_df["user_id"].tolist())

    # --- 1. Gather Discussion-Level Text Metrics ---
    print("Scanning discussion file for user text behavior...")
    disc_text_data = {uid: [] for uid in relevant_users}
    with open(POSTS_DISCUSSION, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                post = json.loads(line)
                uid = str(post.get("user_id", "")).strip()
                if uid in relevant_users:
                    metrics = analyze_text_and_meta(line, CURSE_WORDS)
                    if metrics:
                        disc_text_data[uid].append(metrics)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    # --- 2. Build Enriched Matrices ---
    global_records, disc_records = [], []

    print("Compiling global and discussion matrices...")
    for _, row in tqdm(urls_df.iterrows(), total=len(urls_df), desc="Users Processed"):
        uid = row["user_id"]
        g_urls = row["global_urls"]
        d_urls = row["discussion_urls"]

        # Discussion Profiling
        d_text_metrics = compile_profile_metrics(disc_text_data.get(uid, []))
        d_mbfc_metrics = calculate_mbfc_metrics(d_urls, mbfc_dict)

        # Global Profiling (Reads individual user file)
        g_text_list = []
        g_file = PATH_USER_POSTS / f"{uid}.jsonl"
        if g_file.exists():
            with open(g_file, "r", errors="replace") as f:
                for line in f:
                    metrics = analyze_text_and_meta(line, CURSE_WORDS)
                    if metrics:
                        g_text_list.append(metrics)
        g_text_metrics = compile_profile_metrics(g_text_list)
        g_mbfc_metrics = calculate_mbfc_metrics(g_urls, mbfc_dict)

        # Shared context feature
        inv_ratio = (len(d_urls) / len(g_urls)) if len(g_urls) > 0 else 0.0

        # Assemble Final Records
        d_rec = {"user_id": uid, "discussion_involvement_ratio": inv_ratio}
        d_rec.update(d_text_metrics)
        d_rec.update(d_mbfc_metrics)
        disc_records.append(d_rec)

        g_rec = {"user_id": uid, "discussion_involvement_ratio": inv_ratio}
        g_rec.update(g_text_metrics)
        g_rec.update(g_mbfc_metrics)
        global_records.append(g_rec)

    # Save outputs
    pd.DataFrame(global_records).to_csv(OUTPUT_GLOBAL_CSV, index=False)
    pd.DataFrame(disc_records).to_csv(OUTPUT_DISC_CSV, index=False)
    print(
        f"Extraction complete.\nSaved global features to: {OUTPUT_GLOBAL_CSV.name}\nSaved discussion features to: {OUTPUT_DISC_CSV.name}"
    )


if __name__ == "__main__":
    main()
