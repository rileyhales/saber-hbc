import numpy as np
import pandas as pd

from ._propagation import walk_downstream
from ._propagation import walk_upstream
from ._propagation import propagate_in_table

from ._vocab import model_id_col
from ._vocab import gauge_id_col
from ._vocab import assigned_id_col
from ._vocab import reason_col
from ._vocab import order_col
from ._vocab import area_col


def gauged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns basins a gauge for correction which contain a gauge

    Args:
        df: the assignments table dataframe

    Returns:
        Copy of df with assignments made
    """
    _df = df.copy()
    _df.loc[~_df[gauge_id_col].isna(), assigned_id_col] = _df[gauge_id_col]
    _df.loc[~_df[assigned_id_col].isna(), reason_col] = 'gauged'
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
    for gauged_stream in _df.loc[~_df[gauge_id_col].isna(), model_id_col]:
        connected_segments = walk_downstream(df, gauged_stream, same_order=True)
        _df = propagate_in_table(_df, gauged_stream, connected_segments, max_prop, 'downstream')
        connected_segments = walk_upstream(df, gauged_stream, same_order=True)
        _df = propagate_in_table(_df, gauged_stream, connected_segments, max_prop, 'upstream')

    return _df


def clusters_by_dist(df: pd.DataFrame) -> None:
    """
    Assigns ungauged basins a gauge that is
        (1) is closer than any other gauge
        (2) is of same stream order as ungauged basin
        (3) in the same simulated fdc cluster as ungauged basin

    Args:
        df: the assignments table dataframe

    Returns:
        Copy of df with assignments made
    """
    _df = df.copy()
    # tot_asgn = 0
    # all_ids_assigned = []
    # all_idx_assigned = []
    # first filter by cluster number
    for c_num in sorted(set(_df['sim-fdc-cluster'].values)):
        c_sub = _df[_df['sim-fdc-cluster'] == c_num]
        # next filter by stream order
        for so_num in sorted(set(c_sub['stream_order'])):
            c_so_sub = c_sub[c_sub['stream_order'] == so_num]
            # determine which ids **need** to be assigned
            ids_to_assign = c_so_sub[c_so_sub['assigned_id'].isna()]['model_id'].values
            avail_assigns = c_so_sub[c_so_sub['assigned_id'].notna()]
            if ids_to_assign.size == 0 or avail_assigns.empty:
                print(f'unable to assign cluster {c_num} at stream order {so_num}')
                continue
            # now you find the closest gauge to each unassigned
            for id in ids_to_assign:
                subset = c_so_sub.loc[c_so_sub['model_id'] == id, ['x', 'y']]
                dx = avail_assigns.x.values - subset.x.values
                dy = avail_assigns.y.values - subset.y.values
                avail_assigns['dist'] = np.sqrt(dx * dx + dy * dy)
                row_idx_to_assign = avail_assigns['dist'].idxmin()
                id_to_assign = avail_assigns.loc[row_idx_to_assign].assigned_id
                _df.loc[_df['model_id'] == id, ['assigned_id', 'reason']] = [id_to_assign, f'cluster-{c_num}-dist']
                # all_ids_assigned.append(id)
                # all_idx_assigned.append(row_idx_to_assign)
                # tot_asgn += 1

    return _df


# def clusters(workdir: str, assigns: pd.DataFrame) -> pd.DataFrame:
#     """
#     There is a gauged stream, of the same order as the ungauged, spatially in your cluster. Preference to the station
#     with the closest upstream drainage areas when there are multiple options
#
#     options (pd.DataFrame): a subset of the assignments dataframe which lists only the simulated basins that also
#     contain observation data
#
#     Args:
#         cluster (gpd.GeoDataFrame): the geodataframe of the clustered basins
#         assigns (pd.Dataframe): the dataframe of the assigned and unassigned basins
#
#     """
#     # drop any of the basins that have already been assigned an ID to identify which still need assignment
#     needs_assignment = cluster.drop(cluster[cluster[model_id_col].isin(assigns[assigned_id_col].dropna())].index).values
#     # identify the stream orders in the basins needing to be assigned
#     stream_orders = sorted(set(assigns[assigns[model_id_col].isin(cluster[model_id_col])][order_col].values))
#
#     # for each stream order with unassigned basins
#     for stream_order in stream_orders:
#         # filter a subset dataframe of assignments showing the simulated ID's that contain a station
#         opts_to_assign = assigns[assigns[reason_col] == 'clustered']
#         # the filter the remaining options to only include the stream order we're on
#         opts_to_assign = opts_to_assign[opts_to_assign[order_col] == stream_order]
#
#         # if there were no station points, we cannot apply this option
#         if opts_to_assign.empty:
#             print(f'For stream order {stream_order}, there were no gauges found and so none can be assigned')
#             continue
#
#         # if there is only 1 station option, apply that one to all ungauged basins of the same stream id
#         elif len(opts_to_assign) == 1:
#             ...
#     for i in needs_assignment:
#         assigns.loc[assigns[model_id_col] == i, assigned_id_col] = opts_to_assign[model_id_col].values[0]
#         assigns.loc[assigns[model_id_col] == i, reason_col] = 'SpatiallyClustered'
#
#     # if there are multiple station options, assign the station with the upstream drainage area most similar to the
#     # each of the ungauged basins
#     else:
#         print(opts_to_assign)
#     print(cluster)
#     drain_areas = cluster.merge(opts_to_assign, left_on='COMID', right_on='model_id', how='right')
#     drain_areas = drain_areas[[model_id_col, area_col]]
#     print(drain_areas)
#     for i in needs_assignment:
#         print(i)
#         # find the drainage area of the simulated basin
#         da = cluster[cluster['COMID'] == i]['Tot_Drain_'].values[0]
#         # find the simulated basin which contains a station that has the nearest drainage area
#         closest = opts_to_assign[model_id_col.values[(np.abs(drain_areas['Tot_Drain_'].values - da)).argmin()]
#         # then read what StationID is in the basin that matched by closest drainage area
#         station = opts_to_assign[opts_to_assign[model_id_col] == closest]['StationID'].values[0]
#         # and then finally assign that station id to the ungauged simulated basin
#         print('assigned by spatial cluster and matched by area')
#         assigns.loc[assigns[model_id_col] == i, assigned_id_col] = station
#         assigns.loc[assigns[model_id_col] == i, reason_col] = 'SpatiallyClusteredwArea'
#     return assigns
