import os
from pathlib import Path
from dotenv import load_dotenv

from data_processing.config.constants import USE_TEST_DATA

load_dotenv()


def require_env(key: str) -> str:
    """Validates whether a .env key exists."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


# Paths
REPO_DIR_PATH = Path(require_env("REPO_DIR_PATH"))
DATA_BASE_PATH = Path(require_env("DATA_BASE_PATH"))
DATA_TEST_PATH = Path(require_env("DATA_TEST_PATH"))
USER_POSTS_DIR = Path(require_env("USER_POSTS"))


GENERAL_PATH = Path(require_env("GENERAL_PATH"))
TOPICS_PATH = Path(require_env("TOPICS_PATH"))
HT_CORPORA_PATH = Path(require_env("HT_CORPORA_PATH"))


HT_COUNTS_PATH = Path(require_env("HT_COUNTS_PATH"))
BACKGROUND_CORPUS_PATH = Path(require_env("BACKGROUND_CORPUS_PATH"))

# Derive more paths
DATA_DIR = DATA_TEST_PATH if USE_TEST_DATA else DATA_BASE_PATH
PATH_USER_POSTS = DATA_DIR / USER_POSTS_DIR
