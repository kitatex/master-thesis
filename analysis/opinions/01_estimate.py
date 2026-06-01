import os
import json
import pandas as pd

from config.paths import TOPICS_PATH, MBFC_CSV_PATH

TOPIC_NAME = "nuclearpower"
MIN_RP = 1  # has to match

MIN_URL = 1  # free parameter

# Choose which list of URLs to use for scoring: "global" or "discussion"
URL_SOURCE = "global"

INPUT_CSV = TOPICS_PATH / f"{TOPIC_NAME}/metadata/extracted_urls_minrp{MIN_RP}.csv"

OUTPUT_CSV = (
    TOPICS_PATH
    / f"{TOPIC_NAME}/metadata/opinions_{URL_SOURCE}_minrp{MIN_RP}_minurl{MIN_URL}.csv"
)

BIAS_MAP = {
    "left": -1.5,
    "left-center": -0.5,
    "neutral": 0.0,
    "right-center": 0.5,
    "right": 1.5,
}


def build_mbfc_lookup(mbfc_csv_path):
    """Loads MBFC and creates a quick lookup dictionary for scores."""
    df = pd.read_csv(mbfc_csv_path)
    if "bias_score" not in df.columns:
        df["bias_score"] = df["bias"].map(BIAS_MAP)

    # Drop rows without a valid score
    df = df.dropna(subset=["bias_score"])
    return df.set_index("source")["bias_score"].to_dict()


def calculate_scores():
    print(f"Loading MBFC data from: {MBFC_CSV_PATH}")
    mbfc_lookup = build_mbfc_lookup(MBFC_CSV_PATH)

    print(f"Loading extracted URLs from: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        print(f"Error: Input CSV not found at {INPUT_CSV}")
        return

    df_urls = pd.read_csv(INPUT_CSV)
    records = []

    print(f"Calculating scores based on '{URL_SOURCE}' history...")

    # Dynamically select the column based on the configuration
    target_column = f"{URL_SOURCE}_urls"

    for _, row in df_urls.iterrows():
        user_id = row["user_id"]

        try:
            # Load the JSON string from the chosen column
            urls_to_check = json.loads(row[target_column])
        except (TypeError, json.JSONDecodeError, KeyError):
            urls_to_check = []

        matched_scores = []
        for domain in urls_to_check:
            if domain in mbfc_lookup:
                matched_scores.append(mbfc_lookup[domain])

        if len(matched_scores) >= MIN_URL:
            avg_score = sum(matched_scores) / len(matched_scores)
            records.append(
                {
                    "user_id": user_id,
                    "estimated_opinion": avg_score,
                }
            )

    results_df = pd.DataFrame(records)

    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 40)
    print(f"SCORING COMPLETE ({URL_SOURCE.upper()} URLS)")
    print("=" * 40)
    print(f"Users Meeting Minimum URL Threshold ({MIN_URL}+): {len(results_df)}")
    print(f"Saved Database to: {OUTPUT_CSV}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    calculate_scores()
