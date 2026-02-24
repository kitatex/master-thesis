import csv
import orjson
import time
from pathlib import Path

# --- File Paths ---
INPUT_JSONL = Path(
    "C:/Users/leond/Documents/02-REPOS/06 master thesis/master-thesis/datasets/topics/univie/topics_sie.jsonl"
)


# Define the two output paths
OUTPUT_CSV_OTHER = Path(
    "C:/Users/leond/Documents/02-REPOS/06 master thesis/master-thesis/datasets/topics/univie/topic_posts.csv"
)
OUTPUT_CSV_FAILED_CAMPAIGN = Path(
    "C:/Users/leond/Documents/02-REPOS/06 master thesis/master-thesis/datasets/topics/univie/topic_posts_failed.csv"
)


def convert_jsonl_to_csv_split(
    input_path: Path, output_other_path: Path, output_failed_campaign_path: Path
):
    if not input_path.exists():
        print(f"Error: Could not find input file at {input_path}")
        return

    print(
        f"Loading {input_path.name} into memory as raw bytes... (This will take ~2 seconds)"
    )
    start_time = time.time()

    # Load entire file into memory as raw binary
    with open(input_path, "rb") as f:
        content = f.read()

    print("Data loaded. Slicing file by 'post_id' anchor...")

    # Slice the entire 1.5GB file precisely at the start of every JSON object
    fragments = content.split(b'{"post_id":')

    # The first fragment is just the leading space or '[' before the first post, so we skip it.
    print(f"Found {len(fragments) - 1:,} potential posts! Converting to CSV...")

    with (
        open(output_other_path, "w", newline="", encoding="utf-8") as f_out_other,
        open(
            output_failed_campaign_path, "w", newline="", encoding="utf-8"
        ) as f_out_failed,
    ):
        writer_other = None
        writer_failed = None

        count_other = 0
        count_failed = 0
        errors = 0

        for i in range(1, len(fragments)):
            fragment = fragments[i]

            # Find the very last closing brace '}' in this fragment (ignores commas/brackets after it)
            last_brace_idx = fragment.rfind(b"}")
            if last_brace_idx == -1:
                errors += 1
                continue

            # Reconstruct the perfect, clean JSON byte string
            clean_json = b'{"post_id":' + fragment[: last_brace_idx + 1]

            try:
                data = orjson.loads(clean_json)
            except orjson.JSONDecodeError:
                errors += 1
                continue

            # Initialize BOTH CSV writers using the keys of the first valid object
            if writer_other is None or writer_failed is None:
                headers = list(data.keys())
                writer_other = csv.DictWriter(
                    f_out_other, fieldnames=headers, extrasaction="ignore"
                )
                writer_other.writeheader()
                writer_failed = csv.DictWriter(
                    f_out_failed, fieldnames=headers, extrasaction="ignore"
                )
                writer_failed.writeheader()

            # --- CLEANING LOGIC ---
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    # Convert arrays to strings (e.g. '["eng"]')
                    data[key] = orjson.dumps(value).decode("utf-8")
                elif isinstance(value, str):
                    # Replace line breaks with spaces and delete hidden BOM characters
                    data[key] = (
                        value.replace("\n", " ")
                        .replace("\r", " ")
                        .replace("\ufeff", "")
                    )

            # Check the topic and write to the corresponding CSV
            if data.get("topic") == "failed campaign":
                writer_failed.writerow(data)
                count_failed += 1
            else:
                writer_other.writerow(data)
                count_other += 1

            total_processed = count_other + count_failed
            if total_processed % 50000 == 0:
                print(f"Processed {total_processed:,} posts...")

    elapsed = time.time() - start_time
    print(f"\nConversion complete in {elapsed:.2f} seconds!")
    print(
        f"Saved {count_failed:,} 'failed campaign' posts to: {output_failed_campaign_path.name}"
    )
    print(f"Saved {count_other:,} other posts to: {output_other_path.name}")
    if errors > 0:
        print(f"Note: Skipped {errors:,} unreadable fragments.")


if __name__ == "__main__":
    convert_jsonl_to_csv_split(
        INPUT_JSONL, OUTPUT_CSV_OTHER, OUTPUT_CSV_FAILED_CAMPAIGN
    )
