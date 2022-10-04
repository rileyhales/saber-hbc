import logging
from multiprocessing import Pool

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

__all__ = ['mp_assign_all', 'generate', 'assign_gauged', 'map_propagate', 'map_resolve_props', 'map_assign_ungauged', ]

logger = logging.getLogger(__name__)


def mp_assign_all(workdir: str, assign_df: pd.DataFrame, n_processes: int or None = None) -> pd.DataFrame:
    """

    Args:
        workdir: path to the working directory
        assign_df: the assignments table dataframe with the clustering labels already applied
        n_processes: number of processes to use for multiprocessing, passed to Pool

    Returns:
        pd.DataFrame
    """
    gauged_mids = assign_df[assign_df[gid_col].notna()][mid_col].values

    # assign gauged basins
    logger.info('Assigning Gauged Basins')
    assign_df = assign_gauged(assign_df)

    logger.info('Assigning by Hydraulic Connectivity')
    with Pool(n_processes) as p:
        logger.info('Finding Downstream Assignments')
        df_prop_down = pd.concat(p.starmap(map_propagate, [(assign_df, x, 'down') for x in gauged_mids]))
        logger.info('Caching Downstream Assignments')
        write_table(df_prop_down, workdir, 'prop_downstream')

        logger.info('Finding Upstream Assignments')
        df_prop_up = pd.concat(p.starmap(map_propagate, [(assign_df, x, 'up') for x in gauged_mids]))
        logger.info('Caching Upstream Assignments')
        write_table(df_prop_up, workdir, 'prop_upstream')

        logger.info('Resolving Propagation Assignments')
        df_prop = pd.concat([df_prop_down, df_prop_up]).reset_index(drop=True)
        df_prop = pd.concat(p.starmap(map_resolve_props, [(df_prop, x) for x in df_prop[mid_col].unique()]))
        logger.info('Caching Propagation Assignments')
        write_table(df_prop, workdir, 'prop_resolved')

        logger.info('Assign Remaining Basins by Cluster, Spatial, and Physical Decisions')
        for cluster_number in range(assign_df[clbl_col].max() + 1):
            logger.info(f'Assigning basins in cluster {cluster_number}')
            # limit by cluster number
            c_df = assign_df[assign_df[clbl_col] == cluster_number]
            # keep a list of the unassigned basins in the cluster
            mids = c_df[c_df[reason_col] == 'unassigned'][mid_col].values
            # filter cluster dataframe to find only gauged basins
            c_df = c_df[c_df[gid_col].notna()]
            assign_df = pd.concat([
                pd.concat(p.starmap(map_assign_ungauged, [(assign_df, c_df, x) for x in mids])),
                assign_df[~assign_df[mid_col].isin(mids)]
            ]).reset_index(drop=True)

    logger.info('SABER Assignment Analysis Completed')
    return assign_df


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

    # enforce correct column data types
    labels_df[mid_col] = labels_df[mid_col].astype(str)
    drain_table[mid_col] = drain_table[mid_col].astype(str)
    drain_table[down_mid_col] = drain_table[down_mid_col].astype(str)
    gauge_table[mid_col] = gauge_table[mid_col].astype(str)

    # join the drain_table and gauge_table then join the labels_df
    assign_df = pd.merge(
        drain_table,
        gauge_table,
        on=mid_col,
        how='outer'
    ).merge(labels_df, on=mid_col, how='outer')
    assign_df = assign_df.sort_values(by=mid_col).reset_index(drop=True)

    # create new columns asgn_mid_col, asgn_gid_col, reason_col
    assign_df[[asgn_mid_col, asgn_gid_col, reason_col]] = ['unassigned', 'unassigned', 'unassigned']

    if cache:
        write_table(assign_df, workdir, 'assign_table')
        write_table(assign_df[[mid_col, ]], workdir, 'mid_list')
        write_table(assign_df[[gid_col, ]], workdir, 'gid_list')
        write_table(assign_df[[mid_col, gid_col]], workdir, 'mid_gid_map')

    return assign_df


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
            if stream_row.empty or stream_row[reason_col].values[0] != 'unassigned':
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


def map_resolve_props(df_props: pd.DataFrame, mid: str) -> pd.DataFrame:
    """
    Resolves the propagation assignments by choosing the assignment with the fewest steps

    Args:
        df_props: the combined upstream and downstream propagation assignments dataframe
        mid: the model_id to resolve the propagation assignments for

    Returns:
        pd.DataFrame
    """
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


def map_assign_ungauged(assign_df: pd.DataFrame, gauges_df: np.array, mid: str) -> pd.DataFrame:
    """
    Assigns all possible ungauged basins a gauge that is
        (1) is closer than any other gauge
        (2) is of same stream order as ungauged basin
        (3) in the same simulated fdc cluster as ungauged basin

    Args:
        assign_df: the assignments table dataframe
        gauges_df: a subset of the assignments dataframe containing the gauges
        mid: the model_id to assign a gauge for

    Returns:
        a new row for the given mid with the assignments made
    """
    try:
        # find the closest gauge using euclidean distance without accounting for projection/map distortion
        mid_x, mid_y = assign_df.loc[assign_df[mid_col] == mid, [x_col, y_col]].head(1).values.flatten()
        row_idx_to_assign = pd.Series(
            np.sqrt(np.power(gauges_df[x_col] - mid_x, 2) + np.power(gauges_df[y_col] - mid_y, 2))
        ).idxmin()

        asgn_mid, asgn_gid = gauges_df.loc[row_idx_to_assign, [asgn_mid_col, asgn_gid_col]]
        asgn_reason = f'cluster-{gauges_df[clbl_col].values[0]}'

        new_row = assign_df[assign_df[mid_col] == mid].copy()
        new_row[[asgn_mid_col, asgn_gid_col, reason_col]] = [asgn_mid, asgn_gid, asgn_reason]
    except Exception as e:
        logger.error(f'Error in map_assign_ungauged: {e}')
        new_row = pd.DataFrame(columns=assign_df.columns)

    return new_row
