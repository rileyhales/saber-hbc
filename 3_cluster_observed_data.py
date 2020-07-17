# Adaptation of demo script by: Romain Tavenard
# License: BSD 3 clause
# https://tslearn.readthedocs.io/en/stable/auto_examples/clustering/plot_kmeans.html#sphx-glr-auto-examples-clustering-plot-kmeans-py

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

"""
time_series:
a numpy array where each row is a time series for a single point and there is 1 row for each series

assigned_clusters:
is the same as km.labels_ and shows the cluster number that each row was assigned. Thus it is a list of the same length 
you provided number of input features/rows    

"""

time_series = pd.read_csv('data_1_historical_csv/observed_fdc_normalized.csv', index_col=0).dropna(axis=1)
time_series = np.transpose(time_series.values)
np.random.shuffle(time_series)
time_series = TimeSeriesScalerMeanVariance().fit_transform(time_series)
sz = time_series.shape[1]

km = TimeSeriesKMeans.from_pickle('data_2_cluster_simulations/fdc_eucl_kmeans_16cluster_model.pickle')
n_clusters = km.n_clusters
assigned_clusters_fdc = km.predict(time_series)
print(assigned_clusters_fdc.shape)

fig = plt.figure(figsize=(30, 15), dpi=350)
for yi in range(n_clusters):
    plt.subplot(2, math.ceil(km.n_clusters / 2), yi + 1)
    for xx in time_series[assigned_clusters_fdc == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-3, 3)
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
    if yi == math.floor(n_clusters / 4):
        plt.title("Euclidean $k$-means")

plt.tight_layout()
fig.savefig(f'data_3_cluster_observations/fdc_eucl_kmeans_{n_clusters}cluster.png')

time_series = pd.read_csv('data_1_historical_csv/observed_monavg_normalized.csv', index_col=0).dropna(axis=1)
time_series = np.transpose(time_series.values)
np.random.shuffle(time_series)
time_series = TimeSeriesScalerMeanVariance().fit_transform(time_series)
sz = time_series.shape[1]

km = TimeSeriesKMeans.from_pickle('data_2_cluster_simulations/monavg_eucl_kmeans_16cluster_model.pickle')
n_clusters = km.n_clusters
assigned_clusters_ma = km.predict(time_series)
print(assigned_clusters_ma.shape)

fig = plt.figure(figsize=(30, 15), dpi=350)
for yi in range(n_clusters):
    plt.subplot(2, math.ceil(km.n_clusters / 2), yi + 1)
    for xx in time_series[assigned_clusters_ma == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-3, 3)
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
    if yi == math.floor(n_clusters / 4):
        plt.title("Euclidean $k$-means")

plt.tight_layout()
fig.savefig(f'data_3_cluster_observations/monavg_eucl_kmeans_{n_clusters}cluster.png')

station_ids = pd.read_csv('data_1_historical_csv/observed_fdc_normalized.csv', index_col=0).columns.values
print(station_ids.shape)
print(assigned_clusters_fdc.shape)
print(assigned_clusters_ma.shape)
pd.DataFrame(np.transpose([station_ids, assigned_clusters_fdc, assigned_clusters_ma]),
             columns=('ID', 'obs_ma_cluster', 'obs_fdc_cluster')).to_csv(
    'data_3_cluster_observations/observation_clusters.csv', index=False)
