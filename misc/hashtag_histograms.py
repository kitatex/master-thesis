import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


JSON_FILE = "datasets/general/hashtag_counts_all.json"


# Load occurrence counts
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

data = {key: value for key, value in data.items() if value > 4}

values = pd.Series(data.values(), dtype="int64")

# x = position of the entry in the JSON
x = np.arange(len(values))


fig, ax = plt.subplots(figsize=(14, 7))

ax.scatter(
    x,
    values,
    s=1,
    alpha=0.5,
    rasterized=True,
)

ax.set_yscale("log")

ax.set_xlabel("Hashtags")
ax.set_ylabel("Log Occurrences")
ax.set_title("Distribution of Hashtag Occurrences")

ax.axvline(x=4999, color="red", linewidth=1.5)

# Don't attempt to label 700k x-axis positions
ax.set_xlim(0, len(values) - 1)
ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

ax.grid(
    axis="y",
    which="both",
    alpha=0.2,
)

plt.tight_layout()
plt.show()
