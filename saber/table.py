import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .io import COL_ASN_GID
from .io import COL_ASN_MID
from .io import COL_GID
from .io import COL_GPROP
from .io import COL_MID
from .io import COL_MID_DOWN
from .io import COL_RID
from .io import COL_RPROP
from .io import COL_STRM_ORD
from .io import all_cols
from .io import atable_cols
from .io import atable_cols_defaults
from .io import read_table
from .io import write_table

__all__ = ['init', 'mp_prop_gauges', 'mp_prop_regulated']

logger = logging.getLogger(__name__)


def init(drain_table: pd.DataFrame = None,
         gauge_table: pd.DataFrame = None,
         reg_table: pd.DataFrame = None,
         cluster_table: pd.DataFrame = None,
         cache: bool = True) -> pd.DataFrame:
    """
    Joins the drain_table.csv and gauge_table.csv to create the assign_table.csv

    Args:
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
            drain_table = read_table('drain_table')
        except FileNotFoundError:
            raise FileNotFoundError('The drain_table must be provided or created first')
    if gauge_table is None:
        try:
            gauge_table = read_table('gauge_table')
        except FileNotFoundError:
            raise FileNotFoundError('The gauge_table must be provided or created first')
    if reg_table is None:
        try:
            reg_table = read_table('regulate_table')
        except FileNotFoundError:
            raise FileNotFoundError('The regulate_table must be provided or created first')
    if cluster_table is None:
        try:
            cluster_table = read_table('cluster_table')
        except FileNotFoundError:
            raise FileNotFoundError('The cluster_table must be provided or created first')

    # enforce correct column data types
    drain_table[COL_MID] = drain_table[COL_MID].astype(str)
    drain_table[COL_MID_DOWN] = drain_table[COL_MID_DOWN].astype(str)
    gauge_table[COL_MID] = gauge_table[COL_MID].astype(str)
    gauge_table[COL_GID] = gauge_table[COL_GID].astype(str)
    reg_table[COL_MID] = reg_table[COL_MID].astype(str)
    reg_table[COL_RID] = reg_table[COL_RID].astype(str)
    cluster_table[COL_MID] = cluster_table[COL_MID].astype(str)

    # merge the drain_table, gauge_table, reg_table, and labels_df on the model_id column
    assign_df = (
        drain_table
        .merge(gauge_table, on=COL_MID, how='outer')
        .merge(reg_table, on=COL_MID, how='outer')
        .merge(cluster_table, on=COL_MID, how='outer')
        .sort_values(by=COL_MID)
        .reset_index(drop=True)
    )

    # create new columns asn_mid_col, asn_gid_col, reason_col
    assign_df[atable_cols] = atable_cols_defaults
    assign_df[COL_MID] = assign_df[COL_MID].astype(float).astype(int).astype(str)

    if not all([col in assign_df.columns for col in all_cols]):
        logger.error('Missing columns in assign table. Check your input tables.')
        logger.debug(f'Have columns: {assign_df.columns}')
        logger.debug(f'Need columns: {all_cols}')
        raise AssertionError('Missing columns in assign table. Check your input tables.')

    # check for and remove duplicate rows
    assign_df = assign_df.drop_duplicates(subset=[COL_MID])

    if cache:
        write_table(assign_df, 'assign_table')

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
    logger.info('Propagating from Gauges')
    gauged_mids = df[df[COL_GID].notna()][COL_MID].values

    with Pool(n_processes) as p:
        logger.info('Finding Downstream')
        df_prop_down = pd.concat(p.starmap(_map_propagate, [(df, x, 'down', COL_GPROP) for x in gauged_mids]))
        logger.info('Finding Upstream')
        df_prop_up = pd.concat(p.starmap(_map_propagate, [(df, x, 'up', COL_GPROP) for x in gauged_mids]))
        logger.info('Resolving Nearest Propagation Neighbor')
        df_prop = pd.concat([df_prop_down, df_prop_up]).reset_index(drop=True)
        df_prop = pd.concat(p.starmap(_map_resolve_props, [(df_prop, x, COL_GPROP) for x in df_prop[COL_MID].unique()]))

    return pd.concat([df[~df[COL_MID].isin(df_prop[COL_MID])], df_prop])


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
    with Pool(n_processes) as p:
        logger.info('Propagating Downstream')
        df_prop = pd.concat(p.starmap(
            _map_propagate,
            [(df, x, 'down', COL_RPROP, False) for x in df[df[COL_RID].notna()][COL_MID].values]
        ))
        logger.info('Resolving Propagation')
        df_prop = pd.concat(p.starmap(_map_resolve_props, [(df_prop, x, COL_RPROP) for x in df_prop[COL_MID].unique()]))

    return pd.concat([df[~df[COL_MID].isin(df_prop[COL_MID])], df_prop])


def _map_propagate(df: pd.DataFrame, start_mid: str, direction: str, prop_col: str,
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
    start_row = df[df[COL_MID] == start_mid]
    start_order = start_row[COL_STRM_ORD].values[0]
    start_gid = start_row[COL_ASN_GID].values[0]

    # create a boolean selector array for filtering all future queries
    if same_order:
        select_same_order_streams = True
    else:
        select_same_order_streams = df[COL_STRM_ORD] == start_order

    # # modify the start row to include the propagation information
    # start_row[[COL_ASN_MID, COL_ASN_GID, prop_col]] = [start_mid, start_gid, f'{direction}-{0}-{start_mid}']
    # assigned_rows.append(start_row)

    # counter for the number of steps taken by the loop
    n_steps = 1

    # repeat as long as the current row is not empty
    try:
        while True:
            # select the next up or downstream
            if direction == 'down':
                id_selector = df[COL_MID] == start_row[COL_MID_DOWN].values[0]
            else:  # direction == 'up':
                id_selector = df[COL_MID_DOWN] == start_row[COL_MID].values[0]

            # select the next row using the ID and Order selectors
            start_row = df[np.logical_and(id_selector, select_same_order_streams)]

            # Break the loop if
            # 1. next row is empty -> no upstream reach
            # 2. next row stream order not a match -> not picked by select_same_order_streams -> empty start_row
            if start_row.empty:
                break

            # copy the row, modify the assignment columns, and append to the list
            new_row = start_row.copy()
            new_row[[COL_ASN_MID, COL_ASN_GID, prop_col]] = [start_mid, start_gid, f'{direction}-{n_steps}-{start_mid}']
            assigned_rows.append(new_row)

            # increment the steps counter
            n_steps = n_steps + 1

            # Break the loop if
            # 1. The next row is an outlet -> no downstream row -> cause error when selecting next row
            # 2. we have reach the max number of steps (n_steps -1)
            if int(start_row[COL_MID_DOWN].values[0]) == -1 or n_steps > max_steps:
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
    df = df_props[df_props[COL_MID] == mid].copy()
    # parse the reason statement into number of steps and prop up or downstream
    df[['direction', 'n_steps']] = df[prop_col].apply(lambda x: x.split('-')[:2]).to_list()
    df['n_steps'] = df['n_steps'].astype(int)
    # sort by n_steps then by reason
    df = df.sort_values(['n_steps', 'direction'], ascending=[True, True])
    # return the first row which is the fewest steps and preferring downstream to upstream
    return df.head(1).drop(columns=['direction', 'n_steps'])
