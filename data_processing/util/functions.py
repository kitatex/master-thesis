import re
import orjson
from pathlib import Path
from typing import Generator
from data_processing.config.logging import logger

from data_processing.config.constants import HASHTAG_REGEX


def find_hashtags(text: str) -> list[str]:
    return HASHTAG_REGEX.findall(text)


def extract_hashtags_from_file(file_path: str) -> list[str]:
    """
    Args:
        file_path: The path to the .txt file.

    Returns:
        A list of strings, where each string is a hashtag from the file.
    """

    # Initialize an empty list to store the extracted hashtags
    hashtags = []

    try:
        # Open the file for reading
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Regular expression to find any text enclosed in double quotes (")
        # The pattern '"([^"]*)"' captures the content inside the quotes.
        # It handles the specific format by looking for a pattern like: "hashtag": number
        # even though we only extract the quoted string.
        # The re.findall() function returns a list of all captured groups.
        pattern = r'"([^"]*)"'
        hashtags_words = re.findall(pattern, content)

        # Add a '#' symbol before each hashtag
        hashtags = [f"#{ht}" for ht in hashtags_words]

        print(f"Returning list of {len(hashtags)} hashtags")
        print(f"Preview of list beginning: {hashtags[0:5]}")

        return hashtags

    except FileNotFoundError:
        print(f"Error: The file at path '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def _is_english(post_data: dict) -> bool:
    """
    Standardized check for strictly English posts.
    """
    langs = post_data.get("langs")
    return isinstance(langs, list) and langs == ["eng"]


def get_post_hashtags(data: dict) -> set[str]:
    """
    Returns a set of unique, lowercased hashtags if the post
    passes the language and type filters.
    """
    text = data.get("text", "")

    # Now using the centralized language check!
    if isinstance(text, str) and _is_english(data):
        raw_tags = find_hashtags(text)
        return {tag.lower() for tag in raw_tags}

    return set()


def iter_jsonl(file_path: Path) -> Generator[dict, None, None]:
    """
    Standardized generator to iterate over a JSONL file.
    Handles binary reading, decoding, and basic error logging.
    """
    try:
        with open(file_path, "rb") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield orjson.loads(line)
                except orjson.JSONDecodeError as e:
                    logger.debug(
                        f"JSON parse error in {file_path.name} at line {line_num}: {e}"
                    )
    except IOError as e:
        logger.error(f"Could not read file {file_path}: {e}")
