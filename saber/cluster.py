import glob
import json
import math
import os
from collections.abc import Iterable

import joblib
import kneed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.cluster import KMeans as Clusterer
from sklearn.metrics import silhouette_samples

from ._vocab import cluster_count_file
from ._vocab import mid_col

__all__ = ['generate', 'summarize', 'plot_clusters', 'plot_silhouette', 'merge_assign_table']


def generate(workdir: str, max_clusters: int = 12) -> None:
    """
    Trains kmeans clustering models, saves the model as pickle, generates images and supplementary files

    Args:
        workdir: path to the project directory
        max_clusters: maximum number of clusters to train

    Returns:
        None
    """
    # read the prepared data (array x)
    x = pd.read_parquet(os.path.join(workdir, 'tables', 'hindcast_fdc_transformed.parquet'))
    x = x.values

    # build the kmeans model for a range of cluster numbers
    for n_clusters in range(2, max_clusters + 1):
        print(n_clusters)
        kmeans = Clusterer(n_clusters=n_clusters)
        kmeans.fit_predict(x)
        joblib.dump(kmeans, os.path.join(workdir, 'kmeans_outputs', f'kmeans-{n_clusters}.pickle'))
    return


def summarize(workdir: str) -> None:
    """
    Generate a summary of the clustering results, calculate the silhouette score, save the centers and labels to parquet

    Args:
        workdir: path to the project directory

    Returns:

    """
    summary = {'number': [], 'inertia': [], 'n_iter': [], 'silhouette': []}
    silhouette_scores = []
    labels = []

    # read the prepared data (array x)
    x = pd.read_parquet(os.path.join(workdir, 'tables', 'hindcast_fdc_transformed.parquet'))
    x = x.values

    for model_file in natsorted(glob.glob(os.path.join(workdir, 'kmeans_outputs', 'kmeans-*.pickle'))):
        kmeans = joblib.load(model_file)
        n_clusters = int(kmeans.n_clusters)

        # save the cluster centroids to table - columns are the cluster number, rows are the centroid FDC values
        pd.DataFrame(
            np.transpose(kmeans.cluster_centers_),
            columns=np.array(range(n_clusters)).astype(str)
        ).to_parquet(os.path.join(workdir, 'kmeans_outputs', f'kmeans-{n_clusters}-centers.parquet'))

        # save the silhouette score for each cluster
        silhouette_scores.append(silhouette_samples(x, kmeans.labels_).flatten())
        labels.append(kmeans.labels_.flatten())

        # save the summary stats from this model
        summary['number'].append(n_clusters)
        summary['inertia'].append(kmeans.inertia_)
        summary['n_iter'].append(kmeans.n_iter_)
        summary['silhouette'].append(np.mean(silhouette_scores[-1]))

    # save the summary results as a csv
    pd.DataFrame.from_dict(summary) \
        .to_csv(os.path.join(workdir, 'kmeans_outputs', f'clustering-summary-stats.csv'))
    # save the silhouette scores as a parquet
    silhouette_scores = np.transpose(np.array(silhouette_scores))
    pd.DataFrame(silhouette_scores, columns=np.array(range(2, silhouette_scores.shape[1] + 2)).astype(str)) \
        .to_parquet(os.path.join(workdir, 'kmeans_outputs', 'kmeans-silhouette_scores.parquet'))
    # save the labels as a parquet
    labels = np.transpose(np.array(labels))
    pd.DataFrame(labels, columns=np.array(range(2, labels.shape[1] + 2)).astype(str)) \
        .to_parquet(os.path.join(workdir, 'kmeans_outputs', 'kmeans-labels.parquet'))

    # find the knee/elbow
    knee = kneed.KneeLocator(summary['number'], summary['inertia'], curve='convex', direction='decreasing').knee

    # save the best fitting cluster counts to a csv
    with open(os.path.join(workdir, 'kmeans_outputs', cluster_count_file), 'w') as f:
        f.write(json.dumps({'historical': int(knee)}))
    return


def plot_clusters(workdir: str, clusters: int or Iterable = 'all',
                  max_cols: int = 3, plt_width: int = 3, plt_height: int = 3) -> None:
    """
    Generate figures of the clustered FDC's

    Args:
        workdir: path to the project directory
        clusters: number of clusters to create figures for
        max_cols: maximum number of columns (subplots) in the figure
        plt_width: width of each subplot in inches
        plt_height: height of each subplot in inches

    Returns:
        None
    """
    x = pd.read_parquet(os.path.join(workdir, 'tables', 'hindcast_fdc_transformed.parquet')).values
    size = x.shape[1]
    x_values = np.linspace(0, size, 5)
    x_ticks = np.linspace(0, 100, 5).astype(int)

    kmeans_dir = os.path.join(workdir, 'kmeans_outputs')
    if clusters == 'all':
        model_files = natsorted(glob.glob(os.path.join(kmeans_dir, 'kmeans-*.pickle')))
    elif isinstance(clusters, int):
        model_files = glob.glob(os.path.join(kmeans_dir, f'kmeans-{clusters}.pickle'))
    elif isinstance(clusters, Iterable):
        model_files = natsorted([os.path.join(kmeans_dir, f'kmeans-{i}.pickle') for i in clusters])
    else:
        raise TypeError('n_clusters should be of type int or an iterable')

    for model_file in model_files:
        # todo read the cluster centers from the parquet file instead of the model pickle
        kmeans = joblib.load(model_file)
        n_clusters = int(kmeans.n_clusters)
        n_cols = min(n_clusters, max_cols)
        n_rows = math.ceil(n_clusters / n_cols)

        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(plt_width * n_cols + 1, plt_height * n_rows + 1),
            dpi=500,
            squeeze=False,
            tight_layout=True,
            sharey=True
        )
        fig.suptitle("KMeans FDC Clustering")
        fig.supxlabel('Exceedance Probability (%)')
        fig.supylabel('Discharge Z-Score')

        for i, ax in enumerate(fig.axes[:n_clusters]):
            ax.set_title(f'Cluster {i + 1}')
            ax.set_xlim(0, size)
            ax.set_xticks(x_values, x_ticks)
            ax.set_ylim(-2, 4)
            for j in x[kmeans.labels_ == i]:
                ax.plot(j.ravel(), "k-")
            ax.plot(kmeans.cluster_centers_[i].flatten(), "r-")
        # turn off plotting axes which are blank - made for the square grid but > n_clusters
        for ax in fig.axes[n_clusters:]:
            ax.axis('off')

        fig.savefig(os.path.join(workdir, 'kmeans_outputs', f'kmeans-{n_clusters}.png'))
        plt.close(fig)
    return


def plot_silhouette(workdir: str, clusters: int or Iterable = 'all',
                    plt_width: int = 3, plt_height: int = 3) -> None:
    """
    Plot the silhouette scores for each cluster.
    Based on https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    Args:
        workdir: path to the project directory
        clusters: number of clusters to create figures for
        plt_width: width of each subplot in inches
        plt_height: height of each subplot in inches

    Returns:
        None
    """
    kmeans_dir = os.path.join(workdir, 'kmeans_outputs')
    if clusters == 'all':
        sscore_files = natsorted(glob.glob(os.path.join(kmeans_dir, 'kmeans-*-silscores.parquet')))
    elif isinstance(clusters, int):
        sscore_files = glob.glob(os.path.join(kmeans_dir, f'kmeans-{clusters}-silscores.parquet'))
    elif isinstance(clusters, Iterable):
        sscore_files = natsorted([os.path.join(kmeans_dir, f'kmeans-{i}-silscores.parquet') for i in clusters])
    else:
        raise TypeError('n_clusters should be of type int or an iterable')

    for sscore_file in sscore_files:
        # todo plot the silhouette scores
        n_clusters = int(os.path.basename(sscore_file).split('-')[1])
        centers_df = pd.read_parquet(os.path.join(kmeans_dir, f'kmeans-{n_clusters}-centers.parquet'))
        silscores_df = pd.read_parquet(sscore_file)

        for cluster_num in centers_df.columns:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(
                1,
                2,
                figsize=(plt_width * 2 + 1, plt_height + 1),
                dpi=500,
                squeeze=False,
                tight_layout=True,
            )


def merge_assign_table(workdir: str, assign_table: pd.DataFrame, n_clusters: int = None) -> pd.DataFrame:
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
    # todo move to assign module
    if n_clusters is None:
        # read the cluster results csv
        with open(os.path.join(workdir, 'kmeans_outputs', cluster_count_file), 'r') as f:
            n_clusters = int(json.loads(f.read())['historical'])

    # create a dataframe with the optimal model's labels and the model_id's
    df = pd.DataFrame({
        'cluster': joblib.load(
            os.path.join(workdir, 'kmeans_outputs', f'kmeans-{n_clusters}.pickle')).labels_.flatten(),
        mid_col: pd.read_parquet(os.path.join(workdir, 'tables', 'model_ids.parquet')).values.flatten()
    }, dtype=str)

    # merge the dataframes
    return assign_table.merge(df, how='outer', on=mid_col)
