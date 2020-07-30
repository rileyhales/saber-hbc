import glob
import json
import os
import pprint

import geopandas as gpd
import natsort
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans

data0 = '/Users/riley/code/basin_matching/data_0_inputs'
data1 = '/Users/riley/code/basin_matching/data_1_historical_csv'
data2 = '/Users/riley/code/basin_matching/data_2_clusters'
data3 = '/Users/riley/code/basin_matching/data_3_pairbasins'


def cluster_summary_files(monavg_pickle: str, fdc_pickle: str, name_prefix: str):
    print('Creatings matched csv and jsons')
    # read the label results of the kmeans model previously stored as pickle
    ma_labels = TimeSeriesKMeans.from_pickle(monavg_pickle).labels_.tolist()
    fdc_labels = TimeSeriesKMeans.from_pickle(fdc_pickle).labels_.tolist()
    # create a dataframe showing the comid and assigned cluster number
    ids = pd.read_csv(os.path.join(data1, f'{name_prefix}_fdc_normalized.csv'), index_col=0).dropna(
        axis=1).columns.tolist()
    df = pd.DataFrame(np.transpose([ids, fdc_labels, ma_labels]), columns=('ID', 'fdc_cluster', 'ma_cluster'))
    df.to_csv(os.path.join(data3, f'{name_prefix}_clusters.csv'), index=False)
    # create a json of the paired simulation comids
    df['ma_cluster'] = df['ma_cluster'].astype(int)
    clusters = set(sorted(df['ma_cluster'].values.tolist()))
    pairs = {}
    for i in clusters:
        pairs[i] = df[df['ma_cluster'] == i]['ID'].values.tolist()
    with open(os.path.join(data3, f'{name_prefix}_pairs.json'), 'w') as j:
        j.write(json.dumps(pairs))

    print('Deleting Old GeoJSONs')
    for old in glob.glob(os.path.join(data3, f'{name_prefix}*.geojson')):
        os.remove(old)

    print('Creating GeoJSONs')
    if name_prefix == 'simulated':
        gdf = gpd.read_file(
            os.path.join(data0, 'south_america-geoglows-catchment', 'south_america-geoglows-catchment.shp'))
        # gdf = gpd.read_file(os.path.join(data0, 'south_america-geoglows-drainagline', 'south_america-geoglows-drainagline.shp'))
        for cluster_number in pairs:
            savepath = os.path.join(data3, f'{name_prefix}_cluster_{cluster_number}.geojson')
            gdf[gdf['COMID'].isin(pairs[cluster_number])].to_file(savepath, driver='GeoJSON')
    else:
        gdf = gpd.read_file(os.path.join(data0, 'ideam_stations.json'))
        for cluster_number in pairs:
            savepath = os.path.join(data3, f'{name_prefix}_cluster_{cluster_number}.geojson')
            gdf[gdf['ID'].isin(pairs[cluster_number])].to_file(savepath, driver='GeoJSON')
    return


sim_clusters = 6
obs_clusters = 6
sim_monavg = os.path.join(data2, f'sim_monavg_{sim_clusters}cluster_model.pickle')
sim_fdc = os.path.join(data2, f'sim_fdc_{sim_clusters}cluster_model.pickle')
obs_monavg = os.path.join(data2, f'obs_monavg_{obs_clusters}cluster_model.pickle')
obs_fdc = os.path.join(data2, f'obs_fdc_{obs_clusters}cluster_model.pickle')

print('Simulated Data')
cluster_summary_files(sim_monavg, sim_fdc, 'simulated')
print('Observed Data')
cluster_summary_files(obs_monavg, obs_fdc, 'observed')


print('Identifying spatially paired basins')
sims = tuple(natsort.natsorted(glob.glob(os.path.join(data3, 'simulated*.geojson'))))
obs = tuple(natsort.natsorted(glob.glob(os.path.join(data3, 'observed*.geojson'))))

pairs = {}
for i, sim_cluster_geojson in enumerate(sims):
    print(f'working on sim cluster {i}')
    a = gpd.read_file(sim_cluster_geojson)
    pairs[i] = []
    for j, obs_cluster_geojson in enumerate(obs):
        b = gpd.read_file(obs_cluster_geojson).to_crs(a.crs)
        c = gpd.overlay(b, a, how='intersection')
        if len(c) != 0:
            pairs[i].append(j)
pprint.pprint(pairs)
with open(os.path.join(data3, f'{len(sims)}-{len(obs)}_paired_basins.json'), 'w') as j:
    j.write(json.dumps(pairs))
