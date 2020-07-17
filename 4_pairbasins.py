import numpy as np
import pandas as pd
import json

from tslearn.clustering import TimeSeriesKMeans
monavg_labels = TimeSeriesKMeans.from_pickle(
    'data_2_cluster_simulations/monavg_eucl_kmeans_16cluster_model.pickle').labels_.tolist()
fdc_labels = TimeSeriesKMeans.from_pickle(
    'data_2_cluster_simulations/fdc_eucl_kmeans_16cluster_model.pickle').labels_.tolist()
comids = pd.read_csv('data_1_historical_csv/simulated_fdc_normalized.csv', index_col=0).columns.tolist()
pd.DataFrame(np.transpose([comids, fdc_labels, monavg_labels]), columns=('COMID', 'sim_fdc_cluster', 'sim_ma_cluster'))\
    .to_csv('data_4_pairbasins/simulation_clusters.csv', index=False)


sim = pd.read_csv('data_4_pairbasins/simulation_clusters.csv')
obs = pd.read_csv('data_3_cluster_observations/observation_clusters.csv')
print(sim)
print(obs)

pairs = {}
for i in range(16):
    pairs[i] = {
        'sim': sim[sim['sim_ma_cluster'] == i]['COMID'].values.tolist(),
        'obs': obs[obs['obs_ma_cluster'] == i]['ID'].values.tolist(),
    }
with open('data_4_pairbasins/pairs.json', 'w') as j:
    j.write(json.dumps(pairs))
