import pandas as pd
import igraph as ig
from typing import Iterable, Tuple


def create_weighted_df(interactions_stream: Iterable[Tuple[int, int]]) -> pd.DataFrame:
    """
    Collapses raw interactions into a weighted pandas DataFrame.
    """
    df = pd.DataFrame(interactions_stream, columns=["source", "target"])
    # Remove self-loops
    df = df[df["source"] != df["target"]]

    weighted_df = df.groupby(["source", "target"]).size().reset_index(name="weight")
    return weighted_df


def df_to_igraph(df: pd.DataFrame) -> ig.Graph:
    """
    Converts a weighted dataframe to an igraph object.
    """
    return ig.Graph.TupleList(df.itertuples(index=False), directed=True, weights=True)
