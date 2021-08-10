import glob
import pandas as pd
import geopandas as gpd
import numpy as np


def assign_gauges_to_spatial_clusters(cluster: pd.DataFrame, assigns: pd.DataFrame):
    """
    Scenario 1: SpatiallyClustered
    There is a gauged stream, of the same order as the ungauged, spatially in your cluster. Preference to the station
    with the closest upstream drainage areas when there are multiple options

    options (pd.DataFrame): a subset of the assignments dataframe which lists only the simulated basins that also
    contain observation data

    Args:
        cluster (gpd.GeoDataFrame): the geopandas geodataframe of the clustered basins
        assigns (pd.Dataframe): the dataframe of the assigned and unassigned basins

    """
    # todo use the geojson of observed data points to limit the assignments df to only points within the cluster
    # drop any of the basins that have already been assigned an ID to identify which still need assignment
    needs_assignment = cluster.drop(cluster[cluster['COMID'].isin(assignments['AssignedID'].dropna())].index).values
    # identify the stream orders in the basins needing to be assigned
    stream_orders = sorted(set(assignments[assignments['GeoglowsID'].isin(cluster['COMID'])]['Order'].values))

    # for each stream order with unassigned basins
    for stream_order in stream_orders:
        # filter a subset dataframe of assignments showing the simulated ID's that contain a station
        opts_to_assign = assigns[assigns['AssignmentReason'] == 'spatial']
        # the filter the remaining options to only include the stream order we're on
        opts_to_assign = opts_to_assign[opts_to_assign['Order'] == stream_order]

        # if there were no station points, we cannot apply this option
        if opts_to_assign.empty:
            print(f'For stream order {stream_order}, there were no gauges found and so none can be assigned')
            continue

        # if there is only 1 station option, apply that one to all ungauged basins of the same stream id
        elif len(opts_to_assign) == 1:
            for i in needs_assignment:
                assigns.loc[assigns['GeoglowsID'] == i, 'AssignedID'] = opts_to_assign['COMID'].values[0]
                assigns.loc[assigns['GeoglowsID'] == i, 'AssignmentReason'] = 'SpatiallyClustered'

        # if there are multiple station options, assign the station with the upstream drainage area most similar to the
        # each of the ungauged basins
        else:
            print(opts_to_assign)
            print(cluster)
            drain_areas = cluster.merge(opts_to_assign, left_on='COMID', right_on='GeoglowsID', how='right')
            drain_areas = drain_areas[['GeoglowsID', 'Tot_Drain_']]
            print(drain_areas)
            for i in needs_assignment:
                print(i)
                # find the drainage area of the simulated basin
                da = cluster[cluster['COMID'] == i]['Tot_Drain_'].values[0]
                # find the simulated basin which contains a station that has the nearest drainage area
                closest = opts_to_assign['GeoglowsID'].values[(np.abs(drain_areas['Tot_Drain_'].values - da)).argmin()]
                # then read what StationID is in the basin that matched by closest drainage area
                station = opts_to_assign[opts_to_assign['GeoglowsID'] == closest]['StationID'].values[0]
                # and then finally assign that station id to the ungauged simulated basin
                print('assigned by spatial cluster and matched by area')
                assigns.loc[assigns['GeoglowsID'] == i, 'AssignedID'] = station
                assigns.loc[assigns['GeoglowsID'] == i, 'AssignmentReason'] = 'SpatiallyClusteredwArea'
    return assigns


def assign_intersecting_clusters():
    # Apply option 2: when there is not a gauge of the correct order spatially within a cluster of simulated basins,
    # cluster the gauges and see if there is a gauge of the right order in the cluster of one of the gauges that is
    # spatially within the basin
    return


# todo: apply a scalar from the ratio of the ungauged and gauged upstream areas??
# todo: perform a regression of drainage area v cumulative volume??

# dl = gpd.read_file('data_0_inputs/magdalena_drainagelines/magdalena_drainagelines.shp')
# ctch = gpd.read_file('data_0_inputs/magdalena_catchments/magdalena_catchments.shp')

# identify any of the observation points that are in this cluster (returns a df of points) (should always have 1+)
# obs_pts = gpd.read_file('data_0_inputs/ideam_stations.json').to_crs(epsg=3857)
# obs = gpd.overlay(obs_pts, clusterdf, how='intersection')

assignments = pd.read_csv('../data_4_assignments/AssignmentsTable_modify.csv')

for cluster_geojson in glob.glob('/Users/riley/code/basin_matching/data_3_pairbasins/geojson-sim-6/*.geojson'):
    # read the geojson of the cluster
    clusterdf = gpd.read_file(cluster_geojson)

    # Apply option 1: clustering the data by spatially matching the same order gauges inside the clustered basins
    assignments = assign_gauges_to_spatial_clusters(clusterdf, assignments)
    assignments.to_csv('data_4_assignments/AssignmentsTable_modify_2.csv', index=False)

    # Apply option 2: expanding the search by clustering the gauges

    exit()

    # OPTION 3: AveragedOrder

    # OPTION 4: Cry and go home
