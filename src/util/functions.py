from src.config.constants import HASHTAG_REGEX


def find_hashtags(text: str) -> list[str]:
    return HASHTAG_REGEX.findall(text)
