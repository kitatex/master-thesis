import os
from pathlib import Path
from dotenv import load_dotenv

from src.config.constants import USE_TEST_DATA

load_dotenv()


def require_env(key: str) -> str:
    """Validates whether a .env key exists."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


# Paths
CURRENT_INPUT_PATH = Path(require_env("CURRENT_INPUT_PATH"))
CURRENT_OUTPUT_PATH = Path(require_env("CURRENT_OUTPUT_PATH"))
DATA_BASE_PATH = Path(require_env("DATA_BASE_PATH"))
DATA_TEST_PATH = Path(require_env("DATA_TEST_PATH"))
USER_POSTS_DIR = Path(require_env("USER_POSTS"))

TOPICS_OUTPUT_PATH = Path(require_env("TOPICS_OUTPUT_PATH"))
GENERAL_OUTPUT_PATH = Path(require_env("GENERAL_OUTPUT_PATH"))

HASHTAG_COUNTS_PATH = Path(require_env("HASHTAG_COUNTS_PATH"))

# Derive more paths
DATA_DIR = DATA_TEST_PATH if USE_TEST_DATA else DATA_BASE_PATH
PATH_USER_POSTS = DATA_DIR / USER_POSTS_DIR
