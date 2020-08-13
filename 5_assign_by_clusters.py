import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import natsort

# dl = gpd.read_file('data_0_inputs/magdalena_drainagelines/magdalena_drainagelines.shp')
# ctch = gpd.read_file('data_0_inputs/magdalena_catchments/magdalena_catchments.shp')
obs_pts = gpd.read_file('data_0_inputs/ideam_stations.json').to_crs(epsg=3857)
assigns = pd.read_csv('data_4_assignments/geoglowsID_stationID_assignedID.csv')

for cluster in glob.glob('/Users/riley/code/basin_matching/data_3_pairbasins/geojson-sim-6/*.geojson'):
    # read the geojson of the cluster
    clusterdf = gpd.read_file(cluster)
    # identify any of the observation points that are in this cluster (returns a df of points) (should always have 1+)
    obs = gpd.overlay(obs_pts, clusterdf, how='intersection')
    # drop any of the basins that have already been assigned an ID to identify which still need assignment
    clusterdf.drop(clusterdf[clusterdf['COMID'].isin(assigns['AssignedID'].dropna())].index, inplace=True)
    # identify the stream orders in the basins needing to be assigned
    stream_orders = sorted(set(assigns[assigns['GeoglowsID'].isin(clusterdf['COMID'])]['Order'].values))

    print(clusterdf)

    failed_option1 = []

    # todo: apply a scalar from the ratio of the ungauged and gauged upstream areas??
    # todo: perform a regression of drainage area v cumulative volume??

    # OPTION 1: SpatiallyClustered
    # There is a gauged stream, of the same order as the ungauged, spatially in your cluster. Preference to the station
    # with the closest upstream drainage areas??
    for stream_order in stream_orders:
        # filter the list of options to be only those who were spatially in the cluster -> those in the obs dataframe
        options = assigns[assigns['GeoglowsID'].isin(obs['COMID'])]
        # the filter the remaining options to only include the stream order we're on
        options = options[options['Order'] == stream_order]

        # if there were no observed points, we cannot apply this option
        if options.empty:
            failed_option1.append(stream_order)
            print(f'For stream order {stream_order}, there were no gauges found')
            continue

        # if there is only 1 option, apply that one to them all
        elif len(options) == 1:
            for i in clusterdf:
                assigns.loc[assigns['GeoglowsID'] == i, 'AssignedID'] = options['COMID'].values[0]
                assigns.loc[assigns['GeoglowsID'] == i, 'AssignmentReason'] = 'SpatiallyClustered'

        # if there are multiple options, sort by closest drainage area
        else:
            print(clusterdf['COMID'])
            print(options['GeoglowsID'])
            # print(clusterdf[clusterdf['COMID'] == options['GeoglowsID'].tolist[0]]['Tot_Drain_'])
            drainage_areas = [clusterdf[clusterdf['COMID'] == option]['Tot_Drain_'].values[0]
                              for option in options['GeoglowsID'].tolist()]
            for i in clusterdf:
                da = clusterdf[clusterdf['COMID'] == i]['Tot_Drain_'].values[0]
                station = options[(np.abs(np.array(drainage_areas) - da)).argmin()]
                print(station)
                exit()

            pass

        exit()
        # for option in options:
        # print(options)

    # OPTION 2: Cluster2Cluster
    # There is a gauged stream, of the same order as the ungauged, in the cluster a station spatially in the ungauged
    # cluster. Preference to the station with the closest upstream drainage area and the spatially closest

    # OPTION 3: AveragedOrder

    # OPTION 3: Cry and go home

    # of the remaining catchments, figure out their stream orders
    stream_orders = assigns[assigns['GeoglowsID'].isin(clusterdf['COMID'])]['Order'].values
    print(stream_orders)

