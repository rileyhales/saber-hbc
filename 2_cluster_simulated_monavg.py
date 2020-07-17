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

seed = 0
np.random.seed(seed)
time_series = pd.read_csv('data_1_historical_csv/simulated_monavg_normalized.csv', index_col=0).dropna(axis=1)
time_series = np.transpose(time_series.values)
np.random.shuffle(time_series)
time_series = TimeSeriesScalerMeanVariance().fit_transform(time_series)
sz = time_series.shape[1]

# Euclidean k-means
n_clusters = 16

km = TimeSeriesKMeans(n_clusters=n_clusters, verbose=True, random_state=seed)
assigned_clusters = km.fit_predict(time_series)
km.to_pickle(f'data_2_cluster_simulations/monavg_eucl_kmeans_{n_clusters}cluster_model.pickle')

fig = plt.figure(figsize=(30, 15), dpi=450)
for yi in range(n_clusters):
    plt.subplot(2, math.ceil(n_clusters / 2), yi + 1)
    for xx in time_series[assigned_clusters == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-3, 3)
    plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1), transform=plt.gca().transAxes)
    if yi == math.floor(n_clusters / 4):
        plt.title("Euclidean $k$-means")

plt.tight_layout()
# plt.show()
fig.savefig(f'data_2_cluster_simulations/monavg_eucl_kmeans_{n_clusters}cluster.png')
