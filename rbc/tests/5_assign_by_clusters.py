import glob
import pandas as pd
import geopandas as gpd
import numpy as np





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
    assignments = assign_by_spatial_clusters(clusterdf, assignments)
    assignments.to_csv('data_4_assignments/AssignmentsTable_modify_2.csv', index=False)

    # Apply option 2: expanding the search by clustering the gauges

    exit()

    # OPTION 3: AveragedOrder

    # OPTION 4: Cry and go home
