import math
import os
import json
import glob

import kneed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans as Cluster
from natsort import natsorted

from ._vocab import cluster_count_file
from ._vocab import mid_col


def generate(workdir: str) -> None:
    """
    Trains kmeans clustering models, saves the model as pickle, generates images and supplementary files

    Args:
        workdir: path to the project directory

    Returns:
        None
    """
    inertia = {'number': [], 'inertia': [], 'n_iter': []}

    # read the prepared data (array x)
    x = pd.read_parquet(os.path.join(workdir, 'tables', 'hindcast_fdc_transformed.parquet'))
    x = x.values

    # build the kmeans model for a range of cluster numbers
    for n_clusters in range(1, 17):
        print(n_clusters)
        ks = Cluster(n_clusters=n_clusters, max_iter=150)
        ks.fit_predict(x)
        ks.to_pickle(os.path.join(workdir, 'kmeans_outputs', f'kmeans-{n_clusters}.pickle'))
        inertia['number'].append(n_clusters)
        inertia['inertia'].append(ks.inertia_)
        inertia['n_iter'].append(ks.n_iter_)

    # save the inertia results as a csv
    pd.DataFrame.from_dict(inertia).to_csv(os.path.join(workdir, 'kmeans_outputs', f'cluster-inertia.csv'))

    # find the knee/elbow
    knee = kneed.KneeLocator(inertia['number'], inertia['inertia'], curve='convex', direction='decreasing').knee

    # save the best fitting cluster counts to a csv
    with open(os.path.join(workdir, 'kmeans_outputs', cluster_count_file), 'w') as f:
        f.write(json.dumps({'historical': int(knee)}))
    return


def plot(workdir: str) -> None:
    """
    Generate figures of the clustered FDC's

    Args:
        workdir: path to the project directory

    Returns:
        None
    """
    # image generating params
    img_width = 3
    img_height = 3
    max_cols = 3

    # read the fdc's
    x = pd.read_parquet(os.path.join(workdir, 'tables', 'hindcast_fdc_transformed.parquet')).values
    size = x.shape[1]
    x_values = np.linspace(0, size, 5)
    x_ticks = np.linspace(0, 100, 5).astype(int)

    for model_pickle in natsorted(glob.glob(os.path.join(workdir, 'kmeans_outputs', 'kmeans-*.pickle'))):
        kmeans = Cluster.from_pickle(model_pickle)
        n_clusters = int(kmeans.n_clusters)
        n_cols = min(n_clusters, max_cols)
        n_rows = math.ceil(n_clusters / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(img_width * n_cols + 1, img_height * n_rows + 1), dpi=800,
                                squeeze=False, tight_layout=True, sharey=True)
        fig.suptitle("KMeans FDC Clustering")
        fig.supxlabel('Exceedance Probability (%)')
        fig.supylabel('Discharge Z-Score')

        for i, ax in enumerate(fig.axes[:n_clusters]):
            ax.set_title(f'Cluster {i + 1}')
            ax.set_xlim(0, size)
            ax.set_xticks(x_values, x_ticks)
            ax.set_ylim(-2, 4)
            for j in x[kmeans.labels_ == i]:
                ax.plot(j.ravel(), "k-", alpha=.15)
            ax.plot(kmeans.cluster_centers_[i].flatten(), "r-")
        # turn off plotting axes which are blank - made for the square grid but > n_clusters
        for ax in fig.axes[n_clusters:]:
            ax.axis('off')

        fig.savefig(os.path.join(workdir, 'kmeans_outputs', f'kmeans-{n_clusters}.png'))
        plt.close(fig)
    return


def summarize(workdir: str, assign_table: pd.DataFrame, n_clusters: int = None) -> pd.DataFrame:
    """
    Creates a csv listing the streams assigned to each cluster in workdir/kmeans_models and also adds that information
    to assign_table.csv

    Args:
        workdir: path to the project directory
        assign_table: the assignment table DataFrame
        n_clusters: number of clusters to use when applying the labels to the assign_table

    Returns:
        None
    """
    if n_clusters is None:
        # read the cluster results csv
        with open(os.path.join(workdir, 'kmeans_outputs', cluster_count_file), 'r') as f:
            n_clusters = int(json.loads(f.read())['historical'])

    # create a dataframe with the optimal model's labels and the model_id's
    df = pd.DataFrame({
        'cluster': Cluster.from_pickle(os.path.join(workdir, 'kmeans_outputs', f'kmeans-{n_clusters}.pickle')).labels_.flatten(),
        mid_col: pd.read_parquet(os.path.join(workdir, 'tables', 'model_ids.parquet')).values.flatten()
    }, dtype=str)

    # merge the dataframes
    return assign_table.merge(df, how='outer', on=mid_col)
