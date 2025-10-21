import re
import os

# Data source toggle
USE_TEST_DATA = os.getenv("USE_TEST_DATA", "false").lower() == "true"

# Process
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 8))
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

# For a consistent pattern throughout the program
HASHTAG_REGEX = re.compile(r"#(\w+)")

# From the data/general/ file. Used in tf-idf
TOTAL_POSTS_NUMBER = 233079053
