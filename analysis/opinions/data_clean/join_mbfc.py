import pandas as pd

from config.paths import MBFC_CSV_PATH


PATH_CLEAN = MBFC_CSV_PATH
PATH_RAW = "C:/Users/leond/Documents/02-REPOS/06 master thesis/master-thesis/datasets/url/rating/mbfc_raw.csv"


def enrich_mbfc_data(clean_csv, raw_csv, output_csv):
    print(f"Reading {clean_csv} and {raw_csv}...")

    # Load both datasets
    df_clean = pd.read_csv(clean_csv)
    df_raw = pd.read_csv(raw_csv)

    # Define columns to extract from raw (including the join key)
    meta_cols = [
        "source",
        "country",
        "media_type",
        "popularity",
        "mbfc_credibility_rating",
    ]

    # Ensure we only work with the columns we need and handle potential raw duplicates
    df_meta = df_raw[meta_cols].drop_duplicates(subset=["source"])

    # Perform the Left Join
    # This keeps all rows from 'clean' and adds columns from 'meta' where 'source' matches
    enriched_df = df_clean.merge(df_meta, on="source", how="left")

    # Save the result
    enriched_df.to_csv(output_csv, index=False)

    # Validation stats
    print("\n" + "=" * 30)
    print("ENRICHMENT COMPLETE")
    print("=" * 30)
    print(f"Total entries in clean base: {len(df_clean)}")
    print(
        f"Successfully enriched with metadata: {enriched_df['country'].notna().sum()}"
    )
    print(f"Saved to: {output_csv}")
    print("=" * 30)


if __name__ == "__main__":
    # Update these filenames as needed
    enrich_mbfc_data(PATH_CLEAN, PATH_RAW, "mbfc_clean_extended.csv")
