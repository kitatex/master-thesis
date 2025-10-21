import json

from src.config.paths import CURRENT_INPUT_PATH, CURRENT_OUTPUT_PATH
from src.config.logging import logger

# --- Config ---
threshold = 100

# --- Load JSON file ---
with open(CURRENT_INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Filter dictionary ---
filtered_data = {k: v for k, v in data.items() if v >= threshold}

# --- Save filtered result ---
with open(CURRENT_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

logger.info(f"Filtered dictionary saved to '{CURRENT_OUTPUT_PATH}'.")
logger.info(f"Kept {len(filtered_data)} of {len(data)} keys.")
