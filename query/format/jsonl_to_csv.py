import csv
import orjson
import time
from pathlib import Path

# --- File Paths ---
# Update these to match where your jsonl is saved and where you want the csv
INPUT_JSONL = Path(
    "C:/Users/leond/Documents/02-REPOS/06 master thesis/master-thesis/datasets/topics/colleague_query/topic_posts.jsonl"
)

OUTPUT_CSV = Path(
    "C:/Users/leond/Documents/02-REPOS/06 master thesis/master-thesis/datasets/topics/colleague_query/topic_posts.csv"
)


def convert_jsonl_to_csv(input_path: Path, output_path: Path):
    if not input_path.exists():
        print(f"Error: Could not find input file at {input_path}")
        return

    print(f"Starting conversion of {input_path.name} to CSV...")
    start_time = time.time()

    # Open the input jsonl in read-binary (for orjson) and output csv in write mode
    with (
        open(input_path, "rb") as f_in,
        open(output_path, "w", newline="", encoding="utf-8") as f_out,
    ):
        writer = None

        for line_num, line in enumerate(f_in):
            # Skip empty lines
            if not line.strip():
                continue

            try:
                data = orjson.loads(line)

                # Initialize the CSV writer and write the header using the keys of the first valid line
                if writer is None:
                    # Grab all keys from the JSON object to act as CSV column headers
                    headers = list(data.keys())
                    # extrasaction='ignore' prevents crashing if a post randomly has an extra unexpected key
                    writer = csv.DictWriter(
                        f_out, fieldnames=headers, extrasaction="ignore"
                    )
                    writer.writeheader()

                # CSVs handle simple strings/numbers well, but struggle with lists/dicts like 'langs': ['eng']
                # We convert any lists or dicts back to a JSON string so they fit perfectly in one CSV cell
                for key, value in data.items():
                    if isinstance(value, (list, dict)):
                        data[key] = orjson.dumps(value).decode("utf-8")

                writer.writerow(data)

                # Print progress every 100,000 lines
                if (line_num + 1) % 100000 == 0:
                    print(f"Processed {line_num + 1:,} posts...")

            except orjson.JSONDecodeError:
                print(
                    f"Warning: Could not decode JSON on line {line_num + 1}. Skipping."
                )

    elapsed = time.time() - start_time
    print(f"Conversion complete in {elapsed:.2f} seconds!")
    print(f"CSV saved to: {output_path}")


if __name__ == "__main__":
    convert_jsonl_to_csv(INPUT_JSONL, OUTPUT_CSV)
