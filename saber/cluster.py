import glob
import math
import os
import json

import kneed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from ._vocab import cluster_count_file
from ._vocab import mid_col
from ._vocab import gid_col


def generate(workdir: str) -> None:
    """
    Trains kmeans clustering models, saves the model as pickle, generates images and supplementary files

    Args:
        workdir: path to the project directory

    Returns:
        None
    """
    best_fit = {}

    for table in glob.glob(os.path.join(workdir, 'tables', '*_fdc_*.parquet.gzip')):
        # read the data and transform
        time_series = pd.read_parquet(table)
        time_series = np.transpose(time_series.values)
        time_series = TimeSeriesScalerMeanVariance().fit_transform(time_series)

        # get the name of the data and start tracking stats
        dataset = os.path.basename(table).split('.')[0]
        inertia = {'number': [], 'inertia': [], 'n_iter': []}

        for n_clusters in range(1, 17):
            print(n_clusters)
            # build the kmeans model
            ks = KShape(n_clusters=n_clusters, random_state=0, max_iter=150)
            ks.fit_predict(time_series)
            inertia['number'].append(n_clusters)
            inertia['inertia'].append(ks.inertia_)
            inertia['n_iter'].append(ks.n_iter_)

            # save the trained model
            ks.to_pickle(os.path.join(workdir, 'kmeans_outputs', f'{dataset}-kshape-{n_clusters}_clusters-model.pickle'))

            # generate a plot of the clusters
            size = time_series.shape[1]
            # up to 3 cols, rows determined by number of columns
            n_cols = min(n_clusters, 3)
            n_rows = math.ceil(n_clusters / n_cols)
            img_size = 2.5
            fig = plt.figure(figsize=(img_size * n_cols, img_size * n_rows), dpi=750)

            fig.suptitle("KShape Clustering")
            assigned_clusters = ks.labels_
            for i in range(n_clusters):
                plt.subplot(n_rows, n_cols, i + 1)
                for j in time_series[assigned_clusters == i]:
                    plt.plot(j.ravel(), "k-", alpha=.2)
                plt.plot(ks.cluster_centers_[i].ravel(), "r-")
                plt.xlim(0, size)
                plt.ylim(-2, 4)
                plt.text(0.55, 0.85, f'Cluster {i + 1}', transform=plt.gca().transAxes)
            plt.tight_layout()
            fig.savefig(os.path.join(workdir, 'kmeans_outputs', f'{dataset}-kshape-{n_clusters}_clusters.png'))
            plt.close(fig)

        # save the inertia results as a csv
        pd.DataFrame.from_dict(inertia).to_csv(os.path.join(workdir, 'kmeans_outputs', f'{dataset}-inertia.csv'))
        # find the knee/elbow
        knee = kneed.KneeLocator(inertia['number'], inertia['inertia'], curve='convex', direction='decreasing').knee
        best_fit[dataset] = int(knee)

    # save the best fitting cluster counts to a csv
    with open(os.path.join(workdir, 'kmeans_outputs', cluster_count_file), 'w') as f:
        f.write(json.dumps(best_fit))
    return


def summarize(workdir: str, assign_table: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a csv listing the streams assigned to each cluster in workdir/kmeans_models and also adds that information
    to assign_table.csv

    Args:
        workdir: path to the project directory
        assign_table: the assign_table dataframe

    Returns:
        None
    """
    # read the cluster results csv
    with open(os.path.join(workdir, 'kmeans_models', cluster_count_file), 'r') as f:
        clusters = json.loads(f.read())

    assign_table[mid_col] = assign_table[mid_col].astype(int)

    for dataset, cluster_count in clusters.items():
        # read the list of simulated id's, pair them with their cluster label, save to df
        merge_col = mid_col if "sim" in dataset else gid_col
        csv_path = os.path.join(workdir, f'data_processed', f'{dataset}.csv')
        ids = pd.read_csv(csv_path, index_col=0).columns

        # open the optimal model pickle file
        optimal_model = os.path.join(workdir, 'kmeans_models', f'{dataset}-{cluster_count}-clusters-model.pickle')
        sim_labels = KShape.from_pickle(optimal_model).labels_.tolist()

        # create a dataframe of the ids and their labels (assigned groups)
        df = pd.DataFrame(np.transpose([sim_labels, ids]), columns=[f'{dataset}-cluster', merge_col])
        df.to_csv(os.path.join(workdir, 'kmeans_models', f'optimal-assigns-{dataset}.csv'), index=False)

        # merge the dataframes
        df[merge_col] = df[merge_col].astype(int)
        assign_table = assign_table.merge(df, how='outer', on=merge_col)

    return assign_table
