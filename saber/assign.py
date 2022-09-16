import logging

import numpy as np
import pandas as pd

from .io import asgn_gid_col
from .io import asgn_mid_col
from .io import down_mid_col
from .io import gid_col
from .io import mid_col
from .io import order_col
from .io import read_table
from .io import reason_col
from .io import write_table

__all__ = ['generate', 'assign_gauged', 'map_propagation', 'resolve_propagation', 'assign_by_distance', ]

logger = logging.getLogger(__name__)


def generate(workdir: str, labels_df: pd.DataFrame = None, drain_table: pd.DataFrame = None,
             gauge_table: pd.DataFrame = None, cache: bool = True) -> pd.DataFrame:
    """
    Joins the drain_table.csv and gauge_table.csv to create the assign_table.csv

    Args:
        workdir: path to the working directory
        cache: whether to cache the assign table immediately
        labels_df: a dataframe with a column for the assigned cluster label and a column for the model_id
        drain_table: the drain table dataframe
        gauge_table: the gauge table dataframe

    Returns:
        pd.DataFrame
    """
    # read the tables if they are not provided
    if labels_df is None:
        labels_df = read_table(workdir, 'cluster_labels')
    if drain_table is None:
        drain_table = read_table(workdir, 'drain_table')
    if gauge_table is None:
        gauge_table = read_table(workdir, 'gauge_table')

    labels_df[mid_col] = labels_df[mid_col].astype(str)
    drain_table[mid_col] = drain_table[mid_col].astype(str)
    gauge_table[mid_col] = gauge_table[mid_col].astype(str)

    # join the drain_table and gauge_table then join the labels_df
    assign_table = pd.merge(
        drain_table,
        gauge_table,
        on=mid_col,
        how='outer'
    ).merge(labels_df, on=mid_col, how='outer')

    # create the new columns
    assign_table[asgn_mid_col] = np.nan
    assign_table[asgn_gid_col] = np.nan
    assign_table[reason_col] = np.nan

    if cache:
        write_table(assign_table, workdir, 'assign_table')

    return assign_table


def assign_gauged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns basins a gauge for correction which contain a gauge

    Args:
        df: the assignments table dataframe

    Returns:
        Copy of df1 with assignments made
    """
    selector = df[gid_col].notna()
    df.loc[selector, asgn_mid_col] = df[mid_col]
    df.loc[selector, asgn_gid_col] = df[gid_col]
    df.loc[selector, reason_col] = 'gauged'
    return df


def map_propagation(df: pd.DataFrame, start_mid: int, direction: str) -> pd.DataFrame:
    """
    Meant to be mapped over a dataframe to propagate assignments downstream or upstream

    Args:
        df: the assignments table dataframe
        start_mid: the model_id to start the propagation from
        direction: either 'down' or 'up' to indicate the direction of propagation

    Returns:
        pd.DataFrame
    """
    assigned_rows = []
    start_id_order = df[df[mid_col] == start_mid][order_col].values[0]

    # select the starting row
    stream_row = df[df[mid_col] == start_mid]
    start_gid = stream_row[asgn_gid_col].values[0]
    same_order_streams_selector = df[order_col] == start_id_order

    n_steps = 1

    while len(stream_row):
        if direction == 'down':
            stream_row = df[np.logical_and(
                df[mid_col] == stream_row[down_mid_col].values[0],
                same_order_streams_selector
            )]
        elif direction == 'up':
            stream_row = df[np.logical_and(
                df[down_mid_col] == stream_row[mid_col].values[0],
                same_order_streams_selector
            )]
        else:
            raise ValueError(f'Direction should be "up" or "down", not {direction}')

        new_row = stream_row.copy()
        new_row[[asgn_mid_col, asgn_gid_col, reason_col]] = [start_mid, start_gid, f'prop-{direction}-{n_steps}']
        assigned_rows.append(new_row)

        n_steps += 1

        # repeat while the next downstream is not -1 (outlet)
        if len(stream_row) == 0 or stream_row[down_mid_col].values[0] == -1:
            break
    return pd.concat(assigned_rows)


def resolve_propagation(df: pd.DataFrame, df_prop_down: pd.DataFrame, df_prop_up: pd.DataFrame) -> pd.DataFrame:
    """
    Resolves the propagation assignments by choosing the assignment with the fewest steps

    Args:
        df: the assignments table dataframe
        df_prop_down: the downstream propagation assignments dataframe
        df_prop_up: the upstream propagation assignments dataframe

    Returns:
        pd.DataFrame
    """
    # todo correctly resolve
    # todo log the number of resolved assignments
    df_prop_down = df_prop_down.sort_values(by=mid_col)
    df_prop_up = df_prop_up.sort_values(by=mid_col)

    df.loc[df[mid_col].isin(df_prop_up[mid_col]), [asgn_mid_col, asgn_gid_col, reason_col]] = \
        df_prop_up[[asgn_mid_col, asgn_gid_col, reason_col]].values

    df.loc[df[mid_col].isin(df_prop_down[mid_col]), [asgn_mid_col, asgn_gid_col, reason_col]] = \
        df_prop_down[[asgn_mid_col, asgn_gid_col, reason_col]].values
    return df


def assign_by_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns all possible ungauged basins a gauge that is
        (1) is closer than any other gauge
        (2) is of same stream order as ungauged basin
        (3) in the same simulated fdc cluster as ungauged basin

    Args:
        df: the assignments table dataframe

    Returns:
        df with assignments made
    """
    # first filter by cluster number
    for c_num in sorted(set(df['sim-fdc-cluster'].values)):
        c_sub = df[df['sim-fdc-cluster'] == c_num]
        # next filter by stream order
        for so_num in sorted(set(c_sub[order_col])):
            c_so_sub = c_sub[c_sub[order_col] == so_num]

            # determine which ids **need** to be assigned
            ids_to_assign = c_so_sub[c_so_sub[asgn_mid_col].isna()][mid_col].values
            avail_assigns = c_so_sub[c_so_sub[asgn_mid_col].notna()]
            if ids_to_assign.size == 0 or avail_assigns.empty:
                logger.error(f'unable to assign cluster {c_num} to stream order {so_num}')
                continue
            # now you find the closest gauge to each unassigned
            for mid in ids_to_assign:
                subset = c_so_sub.loc[c_so_sub[mid_col] == mid, ['x', 'y']]

                dx = avail_assigns.x.values - subset.x.values
                dy = avail_assigns.y.values - subset.y.values
                avail_assigns['dist'] = np.sqrt(dx * dx + dy * dy)
                row_idx_to_assign = avail_assigns['dist'].idxmin()

                mid_to_assign = avail_assigns.loc[row_idx_to_assign].assigned_model_id
                gid_to_assign = avail_assigns.loc[row_idx_to_assign].assigned_gauge_id

                df.loc[df[mid_col] == mid, [asgn_mid_col, asgn_gid_col, reason_col]] = \
                    [mid_to_assign, gid_to_assign, f'cluster-{c_num}-dist']

    return df
