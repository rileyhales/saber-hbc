import glob
import json
import math
import os
from collections.abc import Iterable

import joblib
import kneed
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from ._vocab import cluster_count_file

__all__ = ['generate', 'summarize', 'plot_clusters', 'plot_silhouette']


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
        kmeans = KMeans(n_clusters=n_clusters)
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
        print(n_clusters)

        # save the cluster centroids to table - columns are the cluster number, rows are the centroid FDC values
        pd.DataFrame(
            np.transpose(kmeans.cluster_centers_),
            columns=np.array(range(n_clusters)).astype(str)
        ).to_parquet(os.path.join(workdir, 'kmeans_outputs', f'kmeans-centers-{n_clusters}.parquet'))

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
            ax.set_title(f'Cluster {i + 1} (n = {np.sum(kmeans.labels_ == i)})')
            ax.set_xlim(0, size)
            ax.set_xticks(x_values, x_ticks)
            ax.set_ylim(-2, 4)
            for j in x[kmeans.labels_ == i]:
                ax.plot(j.ravel(), "k-")
            ax.plot(kmeans.cluster_centers_[i].flatten(), "r-")
        # turn off plotting axes which are blank (when ax number > n_clusters)
        for ax in fig.axes[n_clusters:]:
            ax.axis('off')

        fig.savefig(os.path.join(workdir, 'kmeans_outputs', f'kmeans-clusters-{n_clusters}.png'))
        plt.close(fig)
    return


def plot_silhouette(workdir: str, plt_width: int = 3, plt_height: int = 3) -> None:
    """
    Plot the silhouette scores for each cluster.
    Based on https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    Args:
        workdir: path to the project directory
        plt_width: width of each subplot in inches
        plt_height: height of each subplot in inches

    Returns:
        None
    """
    labels_df = pd.read_parquet(os.path.join(workdir, 'kmeans_outputs', 'kmeans-labels.parquet'))
    silhouette_df = pd.read_parquet(os.path.join(workdir, 'kmeans_outputs', 'kmeans-silhouette_scores.parquet'))

    for tot_clusters in silhouette_df.columns:
        centers_df = pd.read_parquet(os.path.join(workdir, 'kmeans_outputs', f'kmeans-centers-{tot_clusters}.parquet'))
        fig, (ax1, ax2) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(plt_width * 2 + 1, plt_height + 1),
            dpi=500,
            tight_layout=True,
        )

        # Plot 1 titles and labels
        ax1.set_title("Silhouette Plot")
        ax1.set_xlabel("Silhouette Score")
        ax1.set_ylabel("Cluster Label")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Plot 2 titles and labels
        ax2.set_title("Cluster Centers")
        ax2.set_xlabel("Exceedance Probability (%)")
        ax2.set_ylabel("Discharge Z-Score")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_df[tot_clusters].mean(), color="red", linestyle="--")

        y_lower = 10
        for sub_cluster in range(int(tot_clusters)):
            # select the rows applicable to the current sub cluster
            cluster_silhouettes = silhouette_df[labels_df[tot_clusters] == sub_cluster][tot_clusters].values.flatten()
            cluster_silhouettes.sort()

            n = cluster_silhouettes.shape[0]
            y_upper = y_lower + n

            color = cm.nipy_spectral(sub_cluster / int(tot_clusters))
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouettes,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * n, str(sub_cluster + 1))

            # plot the cluster center
            ax2.plot(centers_df[f'{sub_cluster}'].values, alpha=0.7, c=color, label=f'Cluster {sub_cluster + 1}')

            # add some buffer before the next cluster
            y_lower = y_upper + 10

        fig.savefig(os.path.join(workdir, 'kmeans_outputs', f'kmeans-silhouettes-{tot_clusters}.png'))
        plt.close(fig)
    return
