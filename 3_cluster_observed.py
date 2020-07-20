import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def predict_kmeans_clusters(series: np.array, name: str, n_clusters: int = 12):
    km = TimeSeriesKMeans.from_pickle(f'data_2_cluster_simulations/{name}_eucl_kmeans_{n_clusters}cluster_model.pickle')
    assigned_clusters = km.predict(series)

    sz = series.shape[1]
    fig = plt.figure(figsize=(30, 15), dpi=450)
    for yi in range(n_clusters):
        plt.subplot(2, math.ceil(n_clusters / 2), yi + 1)
        for xx in series[assigned_clusters == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(-3, 3)
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
        if yi == math.floor(n_clusters / 4):
            plt.title("Euclidean $k$-means")

    plt.tight_layout()
    fig.savefig(f'data_3_cluster_observations/{name}_eucl_kmeans_{n_clusters}cluster.png')
    return assigned_clusters


clusters = 12

print('starting fdc')
# predict the observational fdc groups
time_series = pd.read_csv('data_1_historical_csv/observed_fdc_normalized.csv', index_col=0).dropna(axis=1)
time_series = np.transpose(time_series.values)
time_series = TimeSeriesScalerMeanVariance().fit_transform(time_series)
assigned_clusters_fdc = predict_kmeans_clusters(time_series, 'fdc', clusters)

print('starting monavg')
# predict the observational monthly average (seasonality) groups
time_series = pd.read_csv('data_1_historical_csv/observed_monavg_normalized.csv', index_col=0).dropna(axis=1)
time_series = np.transpose(time_series.values)
time_series = TimeSeriesScalerMeanVariance().fit_transform(time_series)
assigned_clusters_ma = predict_kmeans_clusters(time_series, 'monavg', clusters)

# save the clustering assignments to a csv file so we can pair them later
station_ids = pd.read_csv('data_1_historical_csv/observed_fdc_normalized.csv', index_col=0).columns.values
print(station_ids.shape)
print(assigned_clusters_fdc.shape)
print(assigned_clusters_ma.shape)
pd.DataFrame(np.transpose([station_ids, assigned_clusters_fdc, assigned_clusters_ma]),
             columns=('ID', 'obs_ma_cluster', 'obs_fdc_cluster')).to_csv(
    'data_3_cluster_observations/observation_clusters.csv', index=False)
