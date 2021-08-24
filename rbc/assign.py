import numpy as np
import pandas as pd


def gauged(df: pd.DataFrame):
    """

    Args:
        df: the assignments table dataframe

    Returns:

    """
    df.loc[~df['gauge_id'].isna(), 'assigned_id'] = df['gauge_id']
    df.loc[~df['assigned_id'].isna(), 'reason'] = 'gauged'
    return


def propagation(df: pd.DataFrame, station: int, ids: tuple, max_propagation: int = 5) -> pd.DataFrame:
    """

    Args:
        df: the assignments table dataframe
        station: the station to start propagating from
        ids:
        max_propagation:

    Returns:

    """
    for index, i in enumerate(ids):
        distance = index + 1
        if distance > max_propagation:
            print('distance is too large, no longer making assignments')
            continue
        try:
            # if the stream segment has an observation station in it, skip
            if df[df['model_id'] == i]['reason'].values[0] == 'spatial':
                print('found another station, skipping to next station')
                continue
            # if the stream segment does not have an assigned value, assign it
            if pd.isna(df[df['model_id'] == i]['assigned_id'].values[0]):
                df.loc[df['model_id'] == i, 'assigned_id'] = station
                df.loc[df['model_id'] == i, 'reason'] = f'propagation-{distance}'
                print('assigned')
            # if the stream segment does have an assigned value, check if this one is better before overwriting
            else:
                last_dist = int(df[df['model_id'] == i]['reason'].values[0].split('-')[-1])
                print(f'last distance: {last_dist}, new distance: {distance}')
                if distance < int(last_dist):
                    df.loc[df['model_id'] == i, 'assigned_id'] = station
                    df.loc[df['model_id'] == i, 'reason'] = f'propagation-{distance}'
                    print('made better assignment')
                continue
        except Exception as e:
            print(e)
            print(f'failed to set assigned value for geoglows id {i}')
    return df


def assign_by_spatial_clusters(cluster: pd.DataFrame, assigns: pd.DataFrame):
    """
    Scenario 1: SpatiallyClustered
    There is a gauged stream, of the same order as the ungauged, spatially in your cluster. Preference to the station
    with the closest upstream drainage areas when there are multiple options

    options (pd.DataFrame): a subset of the assignments dataframe which lists only the simulated basins that also
    contain observation data

    Args:
        cluster (gpd.GeoDataFrame): the geodataframe of the clustered basins
        assigns (pd.Dataframe): the dataframe of the assigned and unassigned basins

    """
    # todo use the geojson of observed data points to limit the assignments df to only points within the cluster
    # drop any of the basins that have already been assigned an ID to identify which still need assignment
    needs_assignment = cluster.drop(cluster[cluster['COMID'].isin(assigns['assigned_id'].dropna())].index).values
    # identify the stream orders in the basins needing to be assigned
    stream_orders = sorted(set(assigns[assigns['model_id'].isin(cluster['COMID'])]['Order'].values))

    # for each stream order with unassigned basins
    for stream_order in stream_orders:
        # filter a subset dataframe of assignments showing the simulated ID's that contain a station
        opts_to_assign = assigns[assigns['reason'] == 'spatial']
        # the filter the remaining options to only include the stream order we're on
        opts_to_assign = opts_to_assign[opts_to_assign['Order'] == stream_order]

        # if there were no station points, we cannot apply this option
        if opts_to_assign.empty:
            print(f'For stream order {stream_order}, there were no gauges found and so none can be assigned')
            continue

        # if there is only 1 station option, apply that one to all ungauged basins of the same stream id
        elif len(opts_to_assign) == 1:
            for i in needs_assignment:
                assigns.loc[assigns['model_id'] == i, 'assigned_id'] = opts_to_assign['COMID'].values[0]
                assigns.loc[assigns['model_id'] == i, 'reason'] = 'SpatiallyClustered'

        # if there are multiple station options, assign the station with the upstream drainage area most similar to the
        # each of the ungauged basins
        else:
            print(opts_to_assign)
            print(cluster)
            drain_areas = cluster.merge(opts_to_assign, left_on='COMID', right_on='model_id', how='right')
            drain_areas = drain_areas[['model_id', 'Tot_Drain_']]
            print(drain_areas)
            for i in needs_assignment:
                print(i)
                # find the drainage area of the simulated basin
                da = cluster[cluster['COMID'] == i]['Tot_Drain_'].values[0]
                # find the simulated basin which contains a station that has the nearest drainage area
                closest = opts_to_assign['model_id'].values[(np.abs(drain_areas['Tot_Drain_'].values - da)).argmin()]
                # then read what StationID is in the basin that matched by closest drainage area
                station = opts_to_assign[opts_to_assign['model_id'] == closest]['StationID'].values[0]
                # and then finally assign that station id to the ungauged simulated basin
                print('assigned by spatial cluster and matched by area')
                assigns.loc[assigns['model_id'] == i, 'assigned_id'] = station
                assigns.loc[assigns['model_id'] == i, 'reason'] = 'SpatiallyClusteredwArea'
    return assigns


def assign_intersecting_clusters():
    # Apply option 2: when there is not a gauge of the correct order spatially within a cluster of simulated basins,
    # cluster the gauges and see if there is a gauge of the right order in the cluster of one of the gauges that is
    # spatially within the basin
    return
