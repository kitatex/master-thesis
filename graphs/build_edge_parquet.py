import json
import pandas as pd

"""
Parse a JSONL dataset to build a weighted user-to-user edge list
and saves it to a Parquet file.
"""

INPUT_JSONL = "C:/Users/leond/Documents/02-REPOS/06 master thesis/master-thesis/datasets/topics/#climatecrisis/hashtag_corpus/posts_deduplicated.jsonl"
OUTPUT_PARQUET = "C:/Users/leond/Documents/02-REPOS/06 master thesis/master-thesis/datasets/topics/#climatecrisis/graph/climatecrisis_edgelist.parquet"


def build_weighted_edgelist(jsonl_file_path, output_parquet_path):
    edges = []

    print(f"Parsing {jsonl_file_path}...")

    # Step 1: Parse the JSONL file line-by-line for memory efficiency
    with open(jsonl_file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            try:
                post = json.loads(line)

                # The user performing the action (Source)
                source_user = post.get("user_id")

                if source_user is None:
                    continue

                # Check for Reposts
                reposted_author = post.get("reposted_author")
                if reposted_author is not None:
                    edges.append(
                        {
                            "source": source_user,
                            "target": reposted_author,
                            "type": "repost",
                        }
                    )

                # Check for Quotes
                quoted_author = post.get("quoted_author")
                if quoted_author is not None:
                    edges.append(
                        {
                            "source": source_user,
                            "target": quoted_author,
                            "type": "quote",
                        }
                    )

                # Check for Replies (Optional: remove if you only want reposts/quotes)
                replied_author = post.get("replied_author")
                if replied_author is not None:
                    edges.append(
                        {
                            "source": source_user,
                            "target": replied_author,
                            "type": "reply",
                        }
                    )

            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_number}")
                continue

    print(f"Extracted {len(edges)} raw interactions.")

    # Step 2: Convert to a Pandas DataFrame
    df_edges = pd.DataFrame(edges)

    # Drop rows where source and target are the same (self-interactions)
    df_edges = df_edges[df_edges["source"] != df_edges["target"]]

    # Step 3: Calculate Edge Weights
    # Group by source and target, and count the number of occurrences
    print("Calculating edge weights...")
    df_weighted = (
        df_edges.groupby(["source", "target"]).size().reset_index(name="weight")
    )

    print(f"Collapsed into {len(df_weighted)} unique weighted edges.")

    # Step 4: Save to Parquet
    print(f"Saving to {output_parquet_path}...")
    # engine='pyarrow' is standard and usually pre-installed with pandas
    df_weighted.to_parquet(output_parquet_path, engine="pyarrow", index=False)

    print("Done!")
    return df_weighted


# --- Execution ---
if __name__ == "__main__":
    # Replace with your actual file paths

    # Run the pipeline
    weighted_edgelist_df = build_weighted_edgelist(INPUT_JSONL, OUTPUT_PARQUET)

    # Display the first few rows to verify
    print("\nPreview of the Edge List:")
    print(weighted_edgelist_df.head())
