import os

import pandas as pd

__all__ = ['path', 'read', 'cache']


def path(workdir: str) -> str:
    return os.path.join(workdir, 'assign_table.csv')


def read(workdir: str) -> pd.DataFrame:
    return pd.read_csv(path(workdir))


def cache(workdir: str, assign_table: pd.DataFrame) -> None:
    """
    Saves the pandas dataframe to a csv in the proper place in the project directory
    A shortcut for pd.DataFrame.to_csv so you don't have to code it all the time

    Args:
        workdir: the project directory path
        assign_table: the assign_table dataframe

    Returns:
        None
    """
    assign_table.to_csv(path(workdir), index=False)
    return

