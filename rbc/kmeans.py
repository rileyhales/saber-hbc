import math
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def generate_clusters(workdir: str, num_clusters: list = range(4, 13)):
    """
    Creates trained kmeans model pickle files and plots of the results saved as png images

    Args:
        workdir: path to the project directory
        num_clusters: an iterable of integers, the number of kmeans clusters to create.

    Returns:
        None
    """
    tables_to_cluster = glob.glob(os.path.join(workdir, 'data_*', '*.csv'))

    for table in tables_to_cluster:
        # read the data
        time_series = pd.read_csv(table, index_col=0).dropna(axis=1)
        time_series = np.transpose(time_series.values)
        dataset = os.path.basename(table)

        for num_cluster in num_clusters:
            km = TimeSeriesKMeans(n_clusters=num_cluster, verbose=True, random_state=0)
            km.fit_predict(TimeSeriesScalerMeanVariance().fit_transform(time_series))

            # save the trained model
            km.to_pickle(os.path.join(workdir, 'kmeans_models', f'{dataset}-{num_cluster}-clusters-model.pickle'))

            size = time_series.shape[1]
            fig = plt.figure(figsize=(30, 15), dpi=450)
            assigned_clusters = km.labels_
            for i in range(num_cluster):
                plt.subplot(2, math.ceil(num_cluster / 2), i + 1)
                for j in time_series[assigned_clusters == i]:
                    plt.plot(j.ravel(), "k-", alpha=.2)
                plt.plot(km.cluster_centers_[i].ravel(), "r-")
                plt.xlim(0, size)
                plt.ylim(0, np.max(time_series))
                plt.text(0.55, 0.85, f'Cluster {i}', transform=plt.gca().transAxes)
                if i == math.floor(num_cluster / 4):
                    plt.title("Euclidean $k$-means")

            plt.tight_layout()
            fig.savefig(os.path.join(workdir, 'kmeans_images', f'{dataset}-{num_cluster}-clusters.png'))
            plt.close(fig)
    return
