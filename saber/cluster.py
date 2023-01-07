import logging
import math
import os
from collections.abc import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from .io import COL_CID
from .io import COL_MID
from .io import get_dir
from .io import list_cluster_files
from .io import read_table
from .io import write_table

__all__ = [
    'cluster',
    'generate', 'summarize_fit',
    'plot_fit_metrics', 'plot_centers', 'plot_clusters', 'pca_heatmap',
    'predicted_labels_dataframe', 'predict_labels'
]

logger = logging.getLogger(__name__)


def cluster(plot: bool = True) -> None:
    """
    Train k-means cluster models, calculate fit metrics, and generate plots

    Args:
        plot: boolean flag to indicate whether plots should be generated after clustering

    Returns:
        None
    """
    logger.info('Generate Clusters')

    x = read_table("cluster_data").values
    generate(x=x)
    summarize_fit()

    if not plot:
        return

    logger.info('Create Plots')
    plot_centers()
    plot_fit_metrics()
    pca_heatmap()
    return


def generate(x: np.ndarray = None, max_clusters: int = 13) -> None:
    """
    Trains scikit-learn MiniBatchKMeans models and saves as pickle

    Args:
        x: a numpy array of the prepared FDC data
        max_clusters: maximum number of clusters to train

    Returns:
        None
    """
    if x is None:
        x = read_table('cluster_data').values

    # build the kmeans model for a range of cluster numbers
    for n_clusters in range(2, max_clusters + 1):
        logger.info(f'Clustering n={n_clusters}')
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=100)
        kmeans.fit_predict(x)
        joblib.dump(kmeans, os.path.join(get_dir('clusters'), f'kmeans-{n_clusters}.pickle'))
    return


def predict_labels(n_clusters: int, x: pd.DataFrame = None) -> pd.DataFrame:
    """
    Predict the cluster labels for a set number of FDCs

    Args:
        n_clusters: number of cluster model to use for prediction
        x: A dataframe with 1 row per FDC (stream) and 1 column per FDC value. Index is the stream's ID.

    Returns:
        None
    """
    if x is None:
        x = read_table('cluster_data')

    model = joblib.load(os.path.join(get_dir('clusters'), f'kmeans-{n_clusters}.pickle'))
    labels_df = pd.DataFrame(
        np.transpose([model.predict(x.values), x.index]),
        columns=[COL_CID, COL_MID]
    )
    write_table(labels_df, 'cluster_table')
    return labels_df


def predicted_labels_dataframe(x: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load the cluster labels from the parquet file

    Args:
        x: A dataframe with 1 row per FDC (stream) and 1 column per FDC value. Index is the stream's ID.

    Returns:
        A dataframe with 1 row per FDC (stream) and 1 column per cluster label
    """
    # todo use this instead of predict_labels
    if x is None:
        x = read_table('cluster_data')

    df = pd.concat([pd.DataFrame(
        np.transpose(joblib.load(model).predict(x.values)),
        columns=[f'cluster-{os.path.basename(model).split("-")[-1].replace(".pickle", "")}', ],
        index=x.index
    ) for model in list_cluster_files(n_clusters='all')], axis=1)
    write_table(df, 'cluster_labels')
    return df


def summarize_fit() -> None:
    """
    Generate a summary of the clustering results save the centers and labels to parquet

    Returns:
        None
    """
    summary = {'number': [], 'inertia': [], 'n_iter': []}
    labels = []

    for model_file in list_cluster_files(n_clusters='all'):
        logger.info(f'Post Processing {os.path.basename(model_file)}')
        kmeans = joblib.load(model_file)
        n_clusters = int(kmeans.n_clusters)
        labels.append(kmeans.labels_.flatten())

        # save cluster centroids to table - columns are the cluster number, rows are the centroid FDC values
        write_table(
            pd.DataFrame(np.transpose(kmeans.cluster_centers_), columns=np.array(range(n_clusters)).astype(str)),
            f'cluster_centers_{n_clusters}')

        # save the summary stats from this model
        summary['number'].append(n_clusters)
        summary['inertia'].append(kmeans.inertia_)
        summary['n_iter'].append(kmeans.n_iter_)

    # save the summary results as a csv
    sum_df = pd.DataFrame(summary)
    sum_df['knee'] = KneeLocator(summary['number'], summary['inertia'], curve='convex', direction='decreasing').knee
    write_table(sum_df, 'cluster_metrics')

    return


def plot_clusters(x: np.ndarray = None, n_clusters: int or Iterable = 'all',
                  max_cols: int = 3, plt_width: int = 2, plt_height: int = 2, n_lines: int = 2_500) -> None:
    """
    Generate figures of the clustered FDC's

    Args:
        x: a numpy array of the prepared FDC data
        n_clusters: number of clusters to create figures for
        max_cols: maximum number of columns (subplots) in the figure
        plt_width: width of each subplot in inches
        plt_height: height of each subplot in inches
        n_lines: max number of lines to plot in each subplot

    Returns:
        None
    """
    if x is None:
        x = read_table('cluster_data').values

    size = x.shape[1]
    x_values = np.linspace(0, size, 5)
    x_ticks = np.linspace(0, 100, 5).astype(int)

    random_shuffler = np.random.default_rng()

    for model_file in list_cluster_files(n_clusters):
        logger.info(f'Plotting Clusters {os.path.basename(model_file)}')

        # load the model and calculate
        kmeans = joblib.load(model_file)
        n_clusters = int(kmeans.n_clusters)
        n_cols = min(n_clusters, max_cols)
        n_rows = math.ceil(n_clusters / n_cols)

        # initialize the figure and labels
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(plt_width * n_cols + 1, plt_height * n_rows + 1),
            dpi=750,
            squeeze=False,
            tight_layout=True,
            sharey='row'
        )
        fig.suptitle("KMeans FDC Clustering")
        fig.supxlabel('Exceedance Probability (%)')
        fig.supylabel('Discharge Z-Score')

        for i, ax in enumerate(fig.axes[:n_clusters]):
            ax.set_title(f'Cluster {i + 1} (n = {np.sum(kmeans.labels_ == i)})')
            ax.set_xlim(0, size)
            ax.set_xticks(x_values, x_ticks)
            ax.set_ylim(-2, 4)
            fdc_sample = x[kmeans.labels_ == i]
            random_shuffler.shuffle(fdc_sample)
            fdc_sample = fdc_sample[:n_lines]
            for j in fdc_sample:
                ax.plot(j.ravel(), "k-")
            ax.plot(kmeans.cluster_centers_[i].flatten(), "r-")
        # turn off plotting axes which are blank (when ax number > n_clusters)
        for ax in fig.axes[n_clusters:]:
            ax.axis('off')

        fig.savefig(os.path.join(get_dir('clusters'), f'figure-clusters-{n_clusters}.png'))
        plt.close(fig)
    return


def plot_centers(plt_width: int = 2, plt_height: int = 2, max_cols: int = 3) -> None:
    """
    Plot the cluster centers for each cluster.

    Args:
        plt_width: width of each subplot in inches
        plt_height: height of each subplot in inches
        max_cols: maximum number of columns of subplots in the figure

    Returns:
        None
    """
    logger.info('Plotting Cluster Centers')

    clusters_dir = get_dir('clusters')

    for n_clusters in [4, 7, 10, 13]:
        # count number of files to plot
        centers_files = [os.path.join(clusters_dir, f'cluster_centers_{i}.parquet') for i in range(2, n_clusters + 1)]
        n_files = len(centers_files)
        n_cols = min(n_files, max_cols)
        n_rows = math.ceil(n_files / n_cols)

        # initialize the figure and labels
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(plt_width * n_cols + 1.25, plt_height * n_rows + 1.25),
            dpi=750,
            squeeze=False,
            tight_layout=True,
            sharey='row',
            sharex='col'
        )

        fig.suptitle('Cluster Centers', fontsize=16)
        fig.supylabel('Discharge Z-Score')
        fig.supxlabel('Exceedance Probability (%)')

        for centers_table, ax in zip(centers_files, fig.axes[:n_files]):
            n_clusters = int(centers_table.split('_')[-1].split('.')[0])
            centers_df = pd.read_parquet(centers_table, engine='fastparquet')

            for i in range(int(n_clusters)):
                ax.plot(centers_df[f'{i}'].values, label=f'Cluster {i + 1}')

            # Plot titles and labels
            ax.set_title(f"k={n_clusters} clusters")
            ax.set_xlim(0, 40)
            ax.set_ylim(-2, 4)

        fig.savefig(os.path.join(clusters_dir, f'figure-cluster-centers-{n_clusters}.png'))
        plt.close(fig)
    return


def plot_fit_metrics(plt_width: int = 4, plt_height: int = 4) -> None:
    """
    Plot the cluster metrics, inertia and silhouette score, vs number of clusters

    Args:
        plt_width: width of each subplot in inches
        plt_height: height of each subplot in inches

    Returns:
        None
    """
    logger.info('Plotting Cluster Fit Metrics')

    clusters_dir = get_dir('clusters')

    df = read_table('cluster_metrics')

    df['number'] = df['number'].astype(int)
    df['inertia'] = df['inertia'].astype(float)

    # initialize the figure and labels
    fig, ax = plt.subplots(
        figsize=(plt_width, plt_height),
        dpi=750,
        tight_layout=True,
    )

    # Plot titles and labels
    ax.set_title("Clustering Fit Metrics")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")

    ticks = np.arange(1, df['number'].max() + 2)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_xticks(ticks)
    ax.set_yticks([])

    # plot the inertia
    knee = int(df['knee'].values[0])
    ax.plot(df['number'], df['inertia'], marker='o', label='Inertia')
    ax.plot(knee, df[df['number'] == knee]['inertia'], marker='o', c='red', label='Knee')

    fig.savefig(os.path.join(clusters_dir, f'figure-fit-metrics.png'))
    plt.close(fig)
    return


def pca_heatmap(x: pd.DataFrame = None) -> None:
    """
    Plot a heatmap of the principal components

    Args:
        x: the principal components

    Returns:
        None
    """
    if x is None:
        x = read_table('cluster_data')

    logger.info('Plotting PCA Heatmap')

    clusters_dir = get_dir('clusters')

    # calculate the PCA with scikit learn
    pca = PCA(n_components=x.values.shape[1])
    pca.fit(x.values)

    # find the number of components which explain 99.99% of the variance
    # np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.9999) + 1

    # initialize the figure and labels
    fig, ax = plt.subplots(
        figsize=(5, 5),
        dpi=1000,
        tight_layout=True,
    )

    # plot a heatmap for the principal components using seaborn
    sns.heatmap(np.abs(pca.components_.T),
                ax=ax,
                cmap='rocket_r',
                cbar_kws=dict(label='Component Weight Absolute Value'),
                linewidths=0.2, )

    # Plot titles and labels
    fig.suptitle("Principal Components Heatmap")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Feature Number")

    # label the y axis in increments of 5
    ax.set_yticks(np.arange(0, x.values.shape[1], 5))
    ax.set_yticklabels(np.arange(0, x.values.shape[1], 5))

    # label the x axis in increments of 5
    ax.set_xticks(np.arange(0, x.values.shape[1], 5))
    ax.set_xticklabels(np.arange(0, x.values.shape[1], 5))

    fig.savefig(os.path.join(clusters_dir, 'figure_pca_heatmap_1.png'))
    plt.close(fig)
    return
