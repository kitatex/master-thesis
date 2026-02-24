import re

from query.config.constants import HASHTAG_REGEX


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
