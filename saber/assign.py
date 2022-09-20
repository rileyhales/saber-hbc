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
from .io import clbl_col
from .io import x_col
from .io import y_col

__all__ = ['generate', 'assign_gauged', 'map_propagate', 'map_resolve_propagations', 'assign_by_distance', ]

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
    assign_df = pd.merge(
        drain_table,
        gauge_table,
        on=mid_col,
        how='outer'
    ).merge(labels_df, on=mid_col, how='outer')

    # create new columns [asgn_mid_col, asgn_gid_col, reason_col]
    assign_df[[asgn_mid_col, asgn_gid_col, reason_col]] = ['', '', '']
    assign_df[[down_mid_col, mid_col, gid_col]] = assign_df[[down_mid_col, mid_col, gid_col]].astype(str)

    if cache:
        write_table(assign_df, workdir, 'assign_table')

    return assign_df.astype(str)


def assign_gauged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns basins a gauge for correction which contain a gauge

    Args:
        df: the assignments table dataframe

    Returns:
        Copy of df1 with assignments made
    """
    selector = df[gid_col] != 'nan'
    df.loc[selector, asgn_mid_col] = df[mid_col]
    df.loc[selector, asgn_gid_col] = df[gid_col]
    df.loc[selector, reason_col] = 'gauged'
    return df


def map_propagate(df: pd.DataFrame, start_mid: int, direction: str) -> pd.DataFrame or None:
    """
    Meant to be mapped over a dataframe to propagate assignments downstream or upstream

    Args:
        df: the assignments table dataframe
        start_mid: the model_id to start the propagation from
        direction: either 'down' or 'up' to indicate the direction of propagation

    Returns:
        pd.DataFrame
    """
    logger.info(f'Prop {direction} from {start_mid}')
    assigned_rows = []
    start_order = df[df[mid_col] == start_mid][order_col].values[0]

    # select the starting row
    stream_row = df[df[mid_col] == start_mid]
    start_gid = stream_row[asgn_gid_col].values[0]
    select_same_order_streams = df[order_col] == start_order

    n_steps = 1

    # repeat as long as the current row is not empty
    try:
        while True:
            # select the next up or downstream
            if direction == 'down':
                id_selector = df[mid_col] == stream_row[down_mid_col].values[0]
            else:  # direction == 'up':
                id_selector = df[down_mid_col] == stream_row[mid_col].values[0]

            # select the next row using the ID and Order selectors
            stream_row = df[np.logical_and(id_selector, select_same_order_streams)]

            # Break the loop if
            # 1. next row is empty -> no upstream/downstream row -> empty stream_row
            # 2. next row stream order not a match -> not picked by select_same_order_streams -> empty stream_row
            # 3. next row already has an assignment made -> reason column is not empty string
            if stream_row.empty or len(stream_row) == 0 or stream_row[reason_col].values[0] != '':
                break

            # copy the row, modify the assignment columns, and append to the list
            new_row = stream_row.copy()
            new_row[[asgn_mid_col, asgn_gid_col, reason_col]] = [start_mid, start_gid, f'prop-{direction}-{n_steps}']
            assigned_rows.append(new_row)

            # increment the steps counter
            n_steps = n_steps + 1

            # Break the loop if
            # 1. The next row is an outlet -> no downstream row -> cause error when selecting next row
            # 2. we have reach the max number of steps (n_steps -1)
            if stream_row[down_mid_col].values[0] != -1 or n_steps >= 8:
                break
    except Exception as e:
        logger.error(f'Error in map_propagate: {e}')

    if len(assigned_rows):
        return pd.concat(assigned_rows)
    return pd.DataFrame(columns=df.columns)


def map_resolve_propagations(df_props: pd.DataFrame, mid: str) -> pd.DataFrame:
    """
    Resolves the propagation assignments by choosing the assignment with the fewest steps

    Args:
        df_props: the combined upstream and downstream propagation assignments dataframe
        mid: the model_id to resolve the propagation assignments for

    Returns:
        pd.DataFrame
    """
    logger.info(f'Reduce props for {mid}')
    df_mid = df_props[df_props[mid_col] == mid].copy()
    # parse the reason statement into number of steps and prop up or downstream
    df_mid[['direction', 'n_steps']] = pd.DataFrame(
        df_props[reason_col].apply(lambda x: x.split('-')[1:]).to_list())
    df_mid['n_steps'] = df_mid['n_steps'].astype(float)
    # sort by n_steps then by reason
    df_mid = df_mid.sort_values(['n_steps', 'direction'], ascending=[True, True])
    # return the first row which is the fewest steps and preferring downstream to upstream)
    return df_mid.head(1).drop(columns=['direction', 'n_steps'])


def assign_propagation(df: pd.DataFrame, df_props: pd.DataFrame) -> pd.DataFrame:
    """
     Merge the assignment table and the propagation assignments

    Args:
        df: the assignments table dataframe
        df_props: the combined upstream and downstream propagation assignments dataframe

    Returns:
        pd.DataFrame
    """
    return pd.concat([df[~df[mid_col].isin(df_props[mid_col])], df_props])


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
    # todo rewrite this to be parallelizable and run with map like previous functions

    # first filter by cluster number
    for c_num in sorted(df[clbl_col].unique()):
        logger.info(f'Assign by distance for cluster {c_num}')
        c_sub = df[df[clbl_col] == c_num]
        # next filter by stream order
        for so_num in sorted(set(c_sub[order_col])):
            logger.info(f'Stream order {so_num}')
            c_so_sub = c_sub[c_sub[order_col] == so_num]

            # determine which ids **need** to be assigned
            ids_to_assign = c_so_sub[c_so_sub[asgn_mid_col].isna()][mid_col].values
            avail_assigns = c_so_sub[c_so_sub[asgn_mid_col].notna()]
            if ids_to_assign.size == 0 or avail_assigns.empty:
                logger.error(f'unable to assign cluster {c_num} to stream order {so_num}')
                continue
            # now you find the closest gauge to each unassigned
            for mid in ids_to_assign:
                logger.info(f'MID {mid}')
                subset = c_so_sub.loc[c_so_sub[mid_col] == mid, [x_col, y_col]]

                dx = avail_assigns[x_col].values - subset[x_col].values
                dy = avail_assigns[y_col].values - subset[y_col].values
                avail_assigns['dist'] = np.sqrt(dx * dx + dy * dy)
                row_idx_to_assign = avail_assigns['dist'].idxmin()

                mid_to_assign = avail_assigns.loc[row_idx_to_assign].assigned_model_id
                gid_to_assign = avail_assigns.loc[row_idx_to_assign].assigned_gauge_id

                df.loc[df[mid_col] == mid, [asgn_mid_col, asgn_gid_col, reason_col]] = \
                    [mid_to_assign, gid_to_assign, f'cluster-{c_num}-dist']

    return df
