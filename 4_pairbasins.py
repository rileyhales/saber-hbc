import numpy as np
import pandas as pd
import json
from tslearn.clustering import TimeSeriesKMeans


n_clusters = 12
monavg_labels = TimeSeriesKMeans.from_pickle(
    f'data_2_cluster_simulations/monavg_eucl_kmeans_{n_clusters}cluster_model.pickle').labels_.tolist()
fdc_labels = TimeSeriesKMeans.from_pickle(
    f'data_2_cluster_simulations/fdc_eucl_kmeans_{n_clusters}cluster_model.pickle').labels_.tolist()
comids = pd.read_csv('data_1_historical_csv/simulated_fdc_normalized.csv', index_col=0).columns.tolist()
print(len(monavg_labels))
print(len(fdc_labels))
print(len(comids))
pd.DataFrame(np.transpose([comids, fdc_labels, monavg_labels]), columns=('COMID', 'sim_fdc_cluster', 'sim_ma_cluster'))\
    .to_csv('data_4_pairbasins/simulation_clusters.csv', index=False)


sim = pd.read_csv('data_4_pairbasins/simulation_clusters.csv')
obs = pd.read_csv('data_3_cluster_observations/observation_clusters.csv')

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
