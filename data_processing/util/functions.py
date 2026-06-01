import orjson
from pathlib import Path
from typing import Generator
from config.logging import logger

from config.constants import HASHTAG_REGEX


def find_hashtags(text: str) -> list[str]:
    return HASHTAG_REGEX.findall(text)


def extract_hashtags_from_jsonl(file_path: str | Path) -> list[str]:
    """
    Args:
        file_path: Path to the .jsonl (or .json) file.

    Returns:
        A list of hashtags (with leading '#').
    """

    file_path = Path(file_path)

    try:
        with open(file_path, "rb") as f:
            content = f.read()

        data = orjson.loads(content)

        # Case 1: your example (single JSON object / dict)
        if isinstance(data, dict):
            hashtags = [f"#{ht}" for ht in data.keys()]

        # Case 2: true JSONL (one JSON object per line)
        elif isinstance(data, list):
            hashtags = []
            for obj in data:
                if isinstance(obj, dict):
                    hashtags.extend(f"#{ht}" for ht in obj.keys())

        else:
            raise ValueError("Unsupported JSON structure")

        print(f"Returning list of {len(hashtags)} hashtags")
        print(f"Preview of list beginning: {hashtags[:5]}")

        return hashtags

    except FileNotFoundError:
        print(f"Error: The file at path '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def extract_keys_from_jsonl(file_path: str | Path) -> list[str]:
    """
    Args:
        file_path: Path to the .json or .jsonl file.

    Returns:
        A list of extracted keys / values.
    """

    file_path = Path(file_path)

    keys = []

    try:
        with open(file_path, "rb") as f:
            content = f.read()

        # First try parsing as regular JSON
        try:
            data = orjson.loads(content)

            # Case 1: single JSON object / dict
            if isinstance(data, dict):
                keys.extend(data.keys())

            # Case 2: list of JSON objects
            elif isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        keys.extend(obj.keys())

                    elif isinstance(obj, str):
                        keys.append(obj)

            # Case 3: single string
            elif isinstance(data, str):
                keys.append(data)

            else:
                raise ValueError("Unsupported JSON structure")

        # If normal JSON parsing fails, try JSONL
        except orjson.JSONDecodeError:
            with open(file_path, "rb") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    obj = orjson.loads(line)

                    # JSON object
                    if isinstance(obj, dict):
                        keys.extend(obj.keys())

                    # JSON string
                    elif isinstance(obj, str):
                        keys.append(obj)

                    # Optional: support lists inside JSONL
                    elif isinstance(obj, list):
                        for item in obj:
                            if isinstance(item, dict):
                                keys.extend(item.keys())

                            elif isinstance(item, str):
                                keys.append(item)

        print(f"Returning list of {len(keys)} keys")
        print(f"Preview of list beginning: {keys[:5]}")

        return keys

    except Exception as e:
        raise RuntimeError(f"Failed to parse {file_path}: {e}") from e


def optimize_keyword_list(keywords):
    """
    Removes redundant longer keywords if a shorter substring is already in the list.
    Example: ["app", "apple", "resident", "president"] -> ["app", "resident"]
    """
    if not keywords:
        return []

    # 1. Sort by length, shortest strings first
    sorted_kws = sorted(keywords, key=len)
    optimized = []

    for kw in sorted_kws:
        # 2. Check if any already-saved (shorter) keyword is inside this current keyword
        is_redundant = any(saved_kw in kw for saved_kw in optimized)

        # 3. If it's not redundant, add it to our optimized list
        if not is_redundant:
            optimized.append(kw)

    return optimized


def _is_english(post_data: dict) -> bool:
    """
    Standardized check for strictly English posts.
    """
    langs = post_data.get("langs")
    return isinstance(langs, list) and langs == ["eng"]


def get_post_hashtags(data: dict, en_only: bool = False) -> set[str]:
    text = data.get("text", "")

    if isinstance(text, str) and (not en_only or _is_english(data)):
        return {tag.lower() for tag in find_hashtags(text)}

    return set()


def extend_with_no_spaces(strings_list: list[str]) -> list[str]:
    """Extends the input list with versions of its strings that had whitespaces removed."""
    # Create a list of space-removed strings only for those that actually contain spaces
    no_spaces = [s.replace(" ", "") for s in strings_list if " " in s]

    # Extend the original list in-place
    strings_list.extend(no_spaces)

    return strings_list


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
