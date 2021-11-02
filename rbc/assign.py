import numpy as np
import pandas as pd

from ._propagation import walk_downstream
from ._propagation import walk_upstream
from ._propagation import propagate_in_table

from ._vocab import mid_col
from ._vocab import gid_col
from ._vocab import asgn_mid_col
from._vocab import asgn_gid_col
from ._vocab import reason_col
from ._vocab import order_col


def gauged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns basins a gauge for correction which contain a gauge

    Args:
        df: the assignments table dataframe

    Returns:
        Copy of df with assignments made
    """
    _df = df.copy()
    _df.loc[~_df[gid_col].isna(), asgn_mid_col] = _df[mid_col]
    _df.loc[~_df[gid_col].isna(), asgn_gid_col] = _df[gid_col]
    _df.loc[~_df[gid_col].isna(), reason_col] = 'gauged'
    return _df


def propagation(df: pd.DataFrame, max_prop: int = 5) -> pd.DataFrame:
    """

    Args:
        df: the assignments table dataframe
        max_prop: the max number of stream segments to propagate downstream

    Returns:
        Copy of df with assignments made
    """
    _df = df.copy()
    for gauged_stream in _df.loc[~_df[gid_col].isna(), mid_col]:
        if not _df.loc[_df[mid_col] == gauged_stream, gid_col].empty:
            continue
        start_gid = _df.loc[_df[mid_col] == gauged_stream, gid_col].values[0]
        connected_segments = walk_upstream(df, gauged_stream, same_order=True)
        _df = propagate_in_table(_df, gauged_stream, start_gid, connected_segments, max_prop, 'upstream')
        connected_segments = walk_downstream(df, gauged_stream, same_order=True)
        _df = propagate_in_table(_df, gauged_stream, start_gid, connected_segments, max_prop, 'downstream')

    return _df


def clusters_by_monavg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns all possible ungauged basins a gauge that is
        (1) of the same stream order
        (2) in the same simulated fdc cluster
        (3) in the same simulated monavg cluster (monthly averages)
        (4) closest in total drainage area

    This requires matching 2 clusters. Basins in both of the same groups have high likelihood of behaving similarly.

    Args:
        df: the assignments table dataframe

    Returns:
        Copy of df with assignments made
    """
    _df = df.copy()
    return _df


def clusters_by_dist(df: pd.DataFrame) -> pd.DataFrame:
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
    # todo assign based on closest upstream drainage area??
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
