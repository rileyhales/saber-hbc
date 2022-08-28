import os

import joblib
import numpy as np
import pandas as pd

from ._propagation import propagate_in_table
from ._propagation import walk_downstream
from ._propagation import walk_upstream
from ._vocab import asgn_gid_col
from ._vocab import asgn_mid_col
from ._vocab import gid_col
from ._vocab import mid_col
from ._vocab import order_col
from ._vocab import reason_col

__all__ = ['get_path', 'gen', 'read', 'cache', 'merge_clusters', 'assign_gauged', 'assign_propagation',
           'assign_by_distance', ]


def get_path(workdir: str) -> str:
    """
    Creates the path to the assign table based on the working directory provided

    Args:
        workdir: path to the working directory

    Returns:
        absolute path of the assign_table
    """
    return os.path.join(workdir, 'assign_table.parquet.gzip')


def gen(workdir) -> pd.DataFrame:
    """
    Joins the drain_table.csv and gauge_table.csv to create the assign_table.csv

    Args:
        workdir: path to the working directory

    Returns:
        None
    """
    # read and merge the tables
    drain_df = pd.read_parquet(os.path.join(workdir, 'tables', 'drain_table.parquet'))
    gauge_df = pd.read_parquet(os.path.join(workdir, 'tables', 'gauge_table.parquet'))
    assign_table = pd.merge(drain_df, gauge_df, on=mid_col, how='outer')

    # create the new columns
    assign_table[asgn_mid_col] = np.nan
    assign_table[asgn_gid_col] = np.nan
    assign_table[reason_col] = np.nan

    # cache the table
    cache(workdir, assign_table)

    return assign_table


def read(workdir: str) -> pd.DataFrame:
    """
    Reads the assign_table located in the provided directory

    Args:
        workdir: path to the working directory

    Returns:
        assign_table pandas.DataFrame
    """
    return pd.read_parquet(get_path(workdir))


def cache(workdir: str, assign_table: pd.DataFrame) -> None:
    """
    Saves the assignment table dataframe to parquet in the proper place in the project directory so that
    you don't have to code the path every time

    Args:
        workdir: the project directory path
        assign_table: the assign_table dataframe

    Returns:
        None
    """
    assign_table.to_parquet(get_path(workdir))
    return


def merge_clusters(workdir: str, assign_table: pd.DataFrame, n_clusters: int = None) -> pd.DataFrame:
    """
    Creates a csv listing the streams assigned to each cluster in workdir/kmeans_models and also adds that information
    to assign_table.csv

    Args:
        workdir: path to the project directory
        assign_table: the assignment table DataFrame
        n_clusters: number of clusters to use when applying the labels to the assign_table

    Returns:
        None
    """
    # create a dataframe with the optimal model's labels and the model_id's
    df = pd.DataFrame({
        'cluster': joblib.load(os.path.join(workdir, 'clusters', f'kmeans-{n_clusters}.pickle')).labels_,
        mid_col: pd.read_parquet(os.path.join(workdir, 'tables', 'model_ids.parquet')).values.flatten()
    }, dtype=str)

    # merge the dataframes
    return assign_table.merge(df, how='outer', on=mid_col)


def assign_gauged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns basins a gauge for correction which contain a gauge

    Args:
        df: the assignments table dataframe

    Returns:
        Copy of df with assignments made
    """
    _df = df.copy()
    selector = ~_df[gid_col].isna()
    _df.loc[selector, asgn_mid_col] = _df[mid_col]
    _df.loc[selector, asgn_gid_col] = _df[gid_col]
    _df.loc[selector, reason_col] = 'gauged'
    return _df


def assign_propagation(df: pd.DataFrame, max_prop: int = 5) -> pd.DataFrame:
    """

    Args:
        df: the assignments table dataframe
        max_prop: the max number of stream segments to propagate downstream

    Returns:
        Copy of df with assignments made
    """
    _df = df.copy()
    for gauged_stream in _df.loc[~_df[gid_col].isna(), mid_col]:
        subset = _df.loc[_df[mid_col] == gauged_stream, gid_col]
        if subset.empty:
            continue
        start_gid = subset.values[0]
        connected_segments = walk_upstream(df, gauged_stream, same_order=True)
        _df = propagate_in_table(_df, gauged_stream, start_gid, connected_segments, max_prop, 'upstream')
        connected_segments = walk_downstream(df, gauged_stream, same_order=True)
        _df = propagate_in_table(_df, gauged_stream, start_gid, connected_segments, max_prop, 'downstream')

    return _df


def assign_by_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns all possible ungauged basins a gauge that is
        (1) is closer than any other gauge
        (2) is of same stream order as ungauged basin
        (3) in the same simulated fdc cluster as ungauged basin

    Args:
        df: the assignments table dataframe

    Returns:
        Copy of df with assignments made
    """
    _df = df.copy()
    # first filter by cluster number
    for c_num in sorted(set(_df['sim-fdc-cluster'].values)):
        c_sub = _df[_df['sim-fdc-cluster'] == c_num]
        # next filter by stream order
        for so_num in sorted(set(c_sub[order_col])):
            c_so_sub = c_sub[c_sub[order_col] == so_num]

            # determine which ids **need** to be assigned
            ids_to_assign = c_so_sub[c_so_sub[asgn_mid_col].isna()][mid_col].values
            avail_assigns = c_so_sub[c_so_sub[asgn_mid_col].notna()]
            if ids_to_assign.size == 0 or avail_assigns.empty:
                print(f'unable to assign cluster {c_num} to stream order {so_num}')
                continue
            # now you find the closest gauge to each unassigned
            for id in ids_to_assign:
                subset = c_so_sub.loc[c_so_sub[mid_col] == id, ['x', 'y']]

                dx = avail_assigns.x.values - subset.x.values
                dy = avail_assigns.y.values - subset.y.values
                avail_assigns['dist'] = np.sqrt(dx * dx + dy * dy)
                row_idx_to_assign = avail_assigns['dist'].idxmin()

                mid_to_assign = avail_assigns.loc[row_idx_to_assign].assigned_model_id
                gid_to_assign = avail_assigns.loc[row_idx_to_assign].assigned_gauge_id

                _df.loc[_df[mid_col] == id, [asgn_mid_col, asgn_gid_col, reason_col]] = \
                    [mid_to_assign, gid_to_assign, f'cluster-{c_num}-dist']

    return _df
