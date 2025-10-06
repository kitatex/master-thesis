# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# --- Manual ---
# Set keyword(s) to search for (case-insensitive)
KEYWORDS = ["#climatechange"]
# Topic name within data/ dir
OUTPUT_TOPIC = "climatechange"

load_dotenv()


def require_env(key: str) -> str:
    """Validates whether a .env key exists."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


# Data source toggle
USE_TEST_DATA = os.getenv("USE_TEST_DATA", "false").lower() == "true"

# Process
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 8))
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

# Paths
DATA_BASE_PATH = Path(require_env("DATA_BASE_PATH"))
DATA_TEST_PATH = Path(require_env("DATA_TEST_PATH"))
USER_POSTS_DIR = Path(require_env("USER_POSTS"))
TOPICS_OUTPUT_PATH = Path(require_env("TOPICS_OUTPUT_PATH"))

# Derive more paths
DATA_DIR = DATA_TEST_PATH if USE_TEST_DATA else DATA_BASE_PATH
PATH_USER_POSTS = DATA_DIR / USER_POSTS_DIR
OUTPUT_FILE = TOPICS_OUTPUT_PATH / OUTPUT_TOPIC / "posts.jsonl"
