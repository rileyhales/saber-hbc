import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .io import all_cols
from .io import asn_gid_col
from .io import asn_mid_col
from .io import atable_cols
from .io import atable_cols_defaults
from .io import down_mid_col
from .io import gid_col
from .io import gprop_col
from .io import mid_col
from .io import order_col
from .io import read_table
from .io import rid_col
from .io import write_table
from .io import rprop_col

__all__ = ['init', 'mp_prop_gauges', 'mp_prop_regulated']

logger = logging.getLogger(__name__)


def init(workdir: str,
         drain_table: pd.DataFrame = None,
         gauge_table: pd.DataFrame = None,
         reg_table: pd.DataFrame = None,
         cluster_table: pd.DataFrame = None,
         cache: bool = True) -> pd.DataFrame:
    """
    Joins the drain_table.csv and gauge_table.csv to create the assign_table.csv

    Args:
        workdir: path to the working directory
        drain_table: the drain table dataframe
        gauge_table: the gauge table dataframe
        reg_table: the regulatory structure table dataframe
        cluster_table: a dataframe with a column for the assigned cluster label and a column for the model_id
        cache: whether to cache the assign table immediately

    Returns:
        pd.DataFrame
    """
    # read the tables if they are not provided
    if drain_table is None:
        try:
            drain_table = read_table(workdir, 'drain_table')
        except FileNotFoundError:
            raise FileNotFoundError('The drain_table must be provided or created first')
    if gauge_table is None:
        try:
            gauge_table = read_table(workdir, 'gauge_table')
        except FileNotFoundError:
            raise FileNotFoundError('The gauge_table must be provided or created first')
    if reg_table is None:
        try:
            reg_table = read_table(workdir, 'regulate_table')
        except FileNotFoundError:
            raise FileNotFoundError('The regulate_table must be provided or created first')
    if cluster_table is None:
        try:
            cluster_table = read_table(workdir, 'cluster_table')
        except FileNotFoundError:
            raise FileNotFoundError('The cluster_table must be provided or created first')

    # enforce correct column data types
    drain_table[mid_col] = drain_table[mid_col].astype(str)
    drain_table[down_mid_col] = drain_table[down_mid_col].astype(str)
    gauge_table[mid_col] = gauge_table[mid_col].astype(str)
    gauge_table[gid_col] = gauge_table[gid_col].astype(str)
    reg_table[mid_col] = reg_table[mid_col].astype(str)
    reg_table[rid_col] = reg_table[rid_col].astype(str)
    cluster_table[mid_col] = cluster_table[mid_col].astype(str)

    # merge the drain_table, gauge_table, reg_table, and labels_df on the model_id column
    assign_df = (
        drain_table
        .merge(gauge_table, on=mid_col, how='outer')
        .merge(reg_table, on=mid_col, how='outer')
        .merge(cluster_table, on=mid_col, how='outer')
        .sort_values(by=mid_col).reset_index(drop=True)
    )

    # create new columns asn_mid_col, asn_gid_col, reason_col
    assign_df[atable_cols] = atable_cols_defaults

    if not all([col in assign_df.columns for col in all_cols]):
        logger.error('Missing columns in assign table. Check your input tables.')
        logger.debug(f'Have columns: {assign_df.columns}')
        logger.debug(f'Need columns: {all_cols}')
        raise AssertionError('Missing columns in assign table. Check your input tables.')

    if cache:
        write_table(assign_df, workdir, 'assign_table')

    return assign_df


def mp_prop_gauges(df: pd.DataFrame, n_processes: int or None = None) -> pd.DataFrame:
    """
    Traverses dendritic stream networks to identify upstream and downstream river reaches

    Args:
        df: the assign table dataframe
        n_processes: the number of processes to use for multiprocessing

    Returns:
        pd.DataFrame
    """
    logger.info('Propagating from Gauges Basins')
    gauged_mids = df[df[gid_col].notna()][mid_col].values

    with Pool(n_processes) as p:
        logger.info('Finding Downstream Assignments')
        df_prop_down = pd.concat(p.starmap(_map_propagate, [(df, x, 'down', gprop_col) for x in gauged_mids]))
        logger.info('Finding Upstream Assignments')
        df_prop_up = pd.concat(p.starmap(_map_propagate, [(df, x, 'up', gprop_col) for x in gauged_mids]))
        logger.info('Resolving Propagation Assignments')
        df_prop = pd.concat([df_prop_down, df_prop_up]).reset_index(drop=True)
        df_prop = pd.concat(p.starmap(_map_resolve_props, [(df_prop, x, gprop_col) for x in df_prop[mid_col].unique()]))

    return pd.concat([df[~df[mid_col].isin(df_prop[mid_col])], df_prop])


def mp_prop_regulated(df: pd.DataFrame, n_processes: int or None = None) -> pd.DataFrame:
    """
    Traverses dendritic stream networks downstream from regulatory structures

    Args:
        df: the assign table dataframe
        n_processes: the number of processes to use for multiprocessing

    Returns:
        pd.DataFrame
    """
    logger.info('Propagating from Regulatory Structures')
    regulated_ids = df[df[rid_col].notna()][mid_col].values
    with Pool(n_processes) as p:
        logger.info('Propagating Downstream')
        df_prop = pd.concat(p.starmap(_map_propagate, [(df, x, 'down', rprop_col, False) for x in regulated_ids]))
        logger.info('Resolving Propagation')
        df_prop = pd.concat(p.starmap(_map_resolve_props, [(df_prop, x, rprop_col) for x in df_prop[mid_col].unique()]))

    return pd.concat([df[~df[mid_col].isin(df_prop[mid_col])], df_prop])


def _map_propagate(df: pd.DataFrame, start_mid: int, direction: str, prop_col: str,
                   same_order: bool = True, max_steps: int = 15) -> pd.DataFrame or None:
    """
    Meant to be mapped over a dataframe to propagate assignments downstream or upstream

    Args:
        df: the assignments table dataframe
        start_mid: the model_id to start the propagation from
        direction: either 'down' or 'up' to indicate the direction of propagation
        prop_col: the column where the propagation information should be recorded
        max_steps: the maximum number of steps to propagate

    Returns:
        pd.DataFrame
    """
    assigned_rows = []

    # select the row to start the propagation from
    start_order = df[df[mid_col] == start_mid][order_col].values[0]
    stream_row = df[df[mid_col] == start_mid]
    start_gid = stream_row[asn_gid_col].values[0]

    # create a boolean selector array for filtering all future queries
    if same_order:
        select_same_order_streams = True
    else:
        select_same_order_streams = df[order_col] == start_order

    # counter for the number of steps taken by the loop
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
            # 1. next row is empty -> no upstream reach
            # 2. next row stream order not a match -> not picked by select_same_order_streams -> empty stream_row
            if stream_row.empty:
                break

            # copy the row, modify the assignment columns, and append to the list
            new_row = stream_row.copy()
            new_row[[asn_mid_col, asn_gid_col, prop_col]] = [start_mid, start_gid,
                                                             f'prop-{direction}-{n_steps}-{start_mid}']
            assigned_rows.append(new_row)

            # increment the steps counter
            n_steps = n_steps + 1

            # Break the loop if
            # 1. The next row is an outlet -> no downstream row -> cause error when selecting next row
            # 2. we have reach the max number of steps (n_steps -1)
            if int(stream_row[down_mid_col].values[0]) == -1 or n_steps > max_steps:
                break
    except Exception as e:
        logger.error(f'Error in map_propagate: {e}')

    if len(assigned_rows):
        return pd.concat(assigned_rows)
    return pd.DataFrame(columns=df.columns)


def _map_resolve_props(df_props: pd.DataFrame, mid: str, prop_col: str) -> pd.DataFrame:
    """
    Resolves the propagation assignments by choosing the assignment with the fewest steps

    Args:
        df_props: the combined upstream and downstream propagation assignments dataframe
        mid: the model_id to resolve the propagation assignments for
        prop_col: the column where the propagation information is recorded

    Returns:
        pd.DataFrame
    """
    df_mid = df_props[df_props[mid_col] == mid].copy()
    # parse the reason statement into number of steps and prop up or downstream
    df_mid[['direction', 'n_steps']] = pd.DataFrame(df_props[prop_col].apply(lambda x: x.split('-')[1:3]).to_list())
    df_mid['n_steps'] = df_mid['n_steps'].astype(int)
    # sort by n_steps then by reason
    df_mid = df_mid.sort_values(['n_steps', 'direction'], ascending=[True, True])
    # return the first row which is the fewest steps and preferring downstream to upstream
    return df_mid.head(1).drop(columns=['direction', 'n_steps'])
