import json
from pathlib import Path
from typing import Generator, Tuple


def extract_interactions(
    jsonl_path: Path,
    include_reposts: bool = True,
    include_quotes: bool = False,
    include_replies: bool = False,
) -> Generator[Tuple[int, int], None, None]:
    """
    Streams JSONL and yields raw (source, target) user_id tuples.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                post = json.loads(line)
                source = post.get("user_id")
                if source is None:
                    continue

                if include_reposts and post.get("reposted_author"):
                    yield (source, post.get("reposted_author"))

                if include_quotes and post.get("quoted_author"):
                    yield (source, post.get("quoted_author"))

                if include_replies and post.get("replied_author"):
                    yield (source, post.get("replied_author"))
            except json.JSONDecodeError:
                continue
