import glob
import math
import os
import json

import kneed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from .assign import cache_table
from ._vocab import cluster_count_file
from ._vocab import model_id_col


def generate(workdir: str):
    """
    Creates trained kmeans model pickle files and plots of the results saved as png images

    Args:
        workdir: path to the project directory

    Returns:
        None
    """
    best_fit = {}

    for table in glob.glob(os.path.join(workdir, 'data_*', '*fdc.csv')):
        # read the data and transform
        time_series = pd.read_csv(table, index_col=0).dropna(axis=1)
        time_series = np.transpose(time_series.values)
        time_series = TimeSeriesScalerMeanVariance().fit_transform(time_series)

        # get the name of the data and start tracking stats
        dataset = os.path.splitext(os.path.basename(table))[0]
        inertia = {'number': [], 'inertia': []}

        for num_cluster in range(2, 11):
            # build the kmeans model
            km = TimeSeriesKMeans(n_clusters=num_cluster, verbose=True, random_state=0)
            km.fit_predict(time_series)
            inertia['number'].append(num_cluster)
            inertia['inertia'].append(km.inertia_)

            # save the trained model
            km.to_pickle(os.path.join(workdir, 'kmeans_models', f'{dataset}-{num_cluster}-clusters-model.pickle'))

            # generate a plot of the clusters
            size = time_series.shape[1]
            fig = plt.figure(figsize=(30, 15), dpi=450)
            assigned_clusters = km.labels_
            for i in range(num_cluster):
                plt.subplot(2, math.ceil(num_cluster / 2), i + 1)
                for j in time_series[assigned_clusters == i]:
                    plt.plot(j.ravel(), "k-", alpha=.2)
                plt.plot(km.cluster_centers_[i].ravel(), "r-")
                plt.xlim(0, size)
                plt.ylim(-5, 5)
                plt.text(0.55, 0.85, f'Cluster {i}', transform=plt.gca().transAxes)
                if i == math.floor(num_cluster / 4):
                    plt.title("Euclidean $k$-means")
            plt.tight_layout()
            fig.savefig(os.path.join(workdir, 'kmeans_images', f'{dataset}-{num_cluster}-clusters.png'))
            plt.close(fig)

        # save the inertia results as a csv
        pd.DataFrame.from_dict(inertia).to_csv(os.path.join(workdir, 'kmeans_models', f'{dataset}-inertia.csv'))
        # find the knee/elbow
        knee = kneed.KneeLocator(inertia['number'], inertia['inertia'], curve='convex', direction='decreasing').knee
        best_fit[dataset] = int(knee)

    # save the best fitting cluster counts to a csv
    with open(os.path.join(workdir, 'kmeans_models', cluster_count_file), 'w') as f:
        f.write(json.dumps(best_fit))
    return


def summarize(workdir: str):
    """
    Creates a csv listing the streams assigned to each cluster in workdir/kmeans_models and also adds that information
    to assign_table.csv

    Args:
        workdir: path to the project directory

    Returns:
        None
    """
    # read the cluster results csv
    with open(os.path.join(workdir, 'kmeans_models', cluster_count_file), 'r') as f:
        clusters = json.loads(f.read())

    # read the list of simulated id's, pair them with their cluster label, save to df
    sim_ids = pd.read_csv(os.path.join(workdir, 'data_simulated', 'sim-fdc.csv'), index_col=0).columns
    optimal_model = os.path.join(workdir, 'kmeans_models', f'sim-fdc-{clusters["sim-fdc"]}-clusters-model.pickle')
    sim_labels = TimeSeriesKMeans.from_pickle(optimal_model).labels_.tolist()
    sim_df = pd.DataFrame(np.transpose(sim_labels), index=sim_ids, columns=['sim-fdc-cluster', ])
    sim_df.to_csv(os.path.join(workdir, 'kmeans_models', 'optimal-assigns-sim.csv'))

    # read the list of gauge id's, pair them with their cluster label, save to df
    obs_ids = pd.read_csv(os.path.join(workdir, 'data_observed', 'obs-fdc.csv'), index_col=0).columns
    optimal_model = os.path.join(workdir, 'kmeans_models', f'obs-fdc-{clusters["obs-fdc"]}-clusters-model.pickle')
    obs_labels = TimeSeriesKMeans.from_pickle(optimal_model).labels_.tolist()
    obs_df = pd.DataFrame(np.transpose(obs_labels), index=obs_ids, columns=['obs-fdc-cluster', ])
    obs_df.to_csv(os.path.join(workdir, 'kmeans_models', 'optimal-assigns-obs.csv'))

    assign_table = pd.read_csv(os.path.join(workdir, 'assign_table.csv'), index_col=0)
    assign_table[assign_table['model_id']]

    assign_table = assign_table.merge(sim_df, how='inner')
    assign_table = assign_table.merge(sim_df, how='outer', left_index=True, right_index=True)
    cache_table(assign_table, workdir)

    return
