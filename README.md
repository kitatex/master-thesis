# Master Thesis Code

Leon Olszewski

## Setup

- Create a ``uv`` environment based on the ``pyproject.toml``

- Create a ``.env`` file based on the ``.env_template``. In the ``.env`` you set your paths before running individual scripts using ``uv run src.path.to.myscript``.



## Data Processing

Process for obtaining data surrounding a topic:

1. Select a seed hashtag. You can use the hashtag counts in ``data/general/`` to find a seed hashtag.
2. Use ``copy_keyword_posts.py`` to obtain an initial corpus of data of that hashtag.
3. Deduplicate posts using ``deduplicate_posts.py`` because the original data contains duplicates already.
4. Create a idf_score.py for the hashtag


