import pandas as pd

from config.paths import MBFC_CSV_PATH


def inspect_mbfc_duplicates(mbfc_csv_path, output_review_path):
    print(f"Loading MBFC dataset from: {mbfc_csv_path}")
    df = pd.read_csv(mbfc_csv_path)

    # keep=False marks ALL instances of a duplicate as True so you can compare them
    duplicates = df[df.duplicated(subset=["source"], keep=False)]

    if duplicates.empty:
        print("Good news: No duplicates found based on the 'source' column!")
        return

    # Sort by the domain so the duplicated rows sit right next to each other
    duplicates = duplicates.sort_values(by="source")

    num_duplicate_rows = len(duplicates)
    num_unique_domains = duplicates["source"].nunique()

    print("\n" + "=" * 40)
    print("DUPLICATES DETECTED")
    print("=" * 40)
    print(
        f"Found {num_duplicate_rows} total rows representing {num_unique_domains} unique domains."
    )

    # Print a preview to the console
    print("\nPreview of the duplicates:")
    print(duplicates.head(20).to_string(index=False))

    # Save the isolated duplicates to a new CSV for manual review
    duplicates.to_csv(output_review_path, index=False)
    print(
        f"\nAll duplicates have been saved to '{output_review_path}' for your manual review."
    )


if __name__ == "__main__":
    # Update the path to your actual mbfc.csv file
    inspect_mbfc_duplicates(MBFC_CSV_PATH, "mbfc_duplicates_review.csv")
