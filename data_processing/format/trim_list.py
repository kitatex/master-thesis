import json
from pathlib import Path

from data_processing.config.logging import logger

"""
Trim the ranked hashtag list to a certain threshold.
"""

INPUT_PATH = Path("")
OUTPUT_PATH = Path("")

THRESHOLD = 100

# --- Load JSON file ---
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Filter dictionary ---
filtered_data = {k: v for k, v in data.items() if v >= THRESHOLD}

# --- Save filtered result ---
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

logger.info(f"Filtered dictionary saved to '{OUTPUT_PATH}'.")
logger.info(f"Kept {len(filtered_data)} of {len(data)} keys.")
