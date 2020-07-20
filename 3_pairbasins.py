import os
import json
import glob

import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans

data1 = '/Users/riley/code/basin_matching/data_1_historical_csv'
data2 = '/Users/riley/code/basin_matching/data_2_clusters'
data3 = '/Users/riley/code/basin_matching/data_3_cluster_id_pairs'

sim_monavg = glob.glob(os.path.join(data2, 'sim_monavg*.pickle'))[0]
sim_fdc = glob.glob(os.path.join(data2, 'sim_fdc*.pickle'))[0]
obs_monavg = glob.glob(os.path.join(data2, 'obs_monavg*.pickle'))[0]
obs_fdc = glob.glob(os.path.join(data2, 'obs_fdc*.pickle'))[0]

ma_labels = TimeSeriesKMeans.from_pickle(sim_monavg).labels_.tolist()
fdc_labels = TimeSeriesKMeans.from_pickle(sim_fdc).labels_.tolist()
comids = pd.read_csv('data_1_historical_csv/simulated_fdc_normalized.csv', index_col=0).columns.tolist()
# print(len(monavg_labels))
# print(len(fdc_labels))
# print(len(comids))
a = pd.DataFrame(np.transpose([comids, fdc_labels, ma_labels]), columns=('COMID', 'sim_fdc_cluster', 'sim_ma_cluster'))
a.to_csv(os.path.join(data3, 'simulation_clusters.csv'), index=False)

sim = pd.read_csv('data_4_pairbasins/simulation_clusters.csv')
obs = pd.read_csv('data_3_pairbasins/observation_clusters.csv')

# create a json of the paired simulation comids and observed station ids
pairs = {}
for i in range(n_clusters):
    pairs[i] = {
        'sim': sim[sim['sim_ma_cluster'] == i]['COMID'].values.tolist(),
        'obs': obs[obs['obs_ma_cluster'] == i]['ID'].values.tolist(),
    }

# validate that there were no empty lists
for key in pairs:
    if len(pairs[key]['sim']) == 0:
        print(f'Simulation data has no points in cluster {key}')
    if len(pairs[key]['obs']) == 0:
        print(f'Observation data has no points in cluster {key}')

# save the resulting json to a file to be used by the next step
with open('data_4_pairbasins/pairs.json', 'w') as j:
    j.write(json.dumps(pairs))
