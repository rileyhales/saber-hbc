import glob
import logging
import math
import os
from collections.abc import Iterable

import joblib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from natsort import natsorted
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples

from .io import _find_model_files
from .io import read_table
from .io import write_table

__all__ = ['generate', 'summarize_fit', 'plot_clusters', 'calc_silhouette', 'plot_silhouette']

logger = logging.getLogger(__name__)


def generate(workdir: str, x: np.ndarray = None, max_clusters: int = 12) -> None:
    """
    Trains scikit-learn MiniBatchKMeans models and saves as pickle

    Args:
        workdir: path to the project directory
        x: a numpy array of the prepared FDC data
        max_clusters: maximum number of clusters to train

    Returns:
        None
    """
    if x is None:
        x = read_table(workdir, 'hindcast_fdc_trans').values

    # build the kmeans model for a range of cluster numbers
    for n_clusters in range(2, max_clusters + 1):
        logger.info(f'Clustering n={n_clusters}')
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=15)
        kmeans.fit_predict(x)
        joblib.dump(kmeans, os.path.join(workdir, 'clusters', f'kmeans-{n_clusters}.pickle'))
    return


def summarize_fit(workdir: str) -> None:
    """
    Generate a summary of the clustering results save the centers and labels to parquet

    Args:
        workdir: path to the project directory

    Returns:
        None
    """
    summary = {'number': [], 'inertia': [], 'n_iter': []}
    labels = []

    for model_file in _find_model_files(workdir, n_clusters='all'):
        logger.info(f'Summarizing {os.path.basename(model_file)}')
        kmeans = joblib.load(model_file)
        n_clusters = int(kmeans.n_clusters)
        labels.append(kmeans.labels_.flatten())

        # save cluster centroids to table - columns are the cluster number, rows are the centroid FDC values
        write_table(
            pd.DataFrame(np.transpose(kmeans.cluster_centers_), columns=np.array(range(n_clusters)).astype(str)),
            workdir,
            f'cluster_centers_{n_clusters}'
        )

        # save the summary stats from this model
        summary['number'].append(n_clusters)
        summary['inertia'].append(kmeans.inertia_)
        summary['n_iter'].append(kmeans.n_iter_)

    # save the summary results as a csv
    sum_df = pd.DataFrame(summary)
    sum_df['knee'] = KneeLocator(summary['number'], summary['inertia'], curve='convex', direction='decreasing').knee
    write_table(sum_df, workdir, 'cluster_metrics')

    # save the labels as a parquet
    labels = np.transpose(np.array(labels))
    write_table(pd.DataFrame(labels, columns=np.array(range(2, labels.shape[1] + 2)).astype(str)),
                workdir, 'cluster_labels')
    return


def calc_silhouette(workdir: str, x: np.ndarray, n_clusters: int or Iterable = 'all',
                    samples: int = 5e5) -> None:
    """
    Calculate the silhouette score for the given number of clusters

    Args:
        workdir: path to the project directory
        x: a numpy array of the prepared FDC data
        n_clusters: the number of clusters to calculate the silhouette score for
        samples: the number of samples to use for the silhouette score calculation

    Returns:
        None
    """
    if x is None:
        x = read_table(workdir, 'hindcast_fdc_trans').values
    fdc_df = pd.DataFrame(x)

    summary = {'number': [], 'silhouette': []}

    random_shuffler = np.random.default_rng()

    for model_file in _find_model_files(workdir, n_clusters):
        logger.info(f'Calculating silhouette for {os.path.basename(model_file)}')
        kmeans = joblib.load(model_file)

        # randomly sample fdcs from each cluster
        fdc_df['label'] = kmeans.labels_
        ss_df = pd.DataFrame(columns=fdc_df.columns.to_list())
        for i in range(int(kmeans.n_clusters)):
            values = fdc_df[fdc_df['label'] == i].drop(columns='label').values
            random_shuffler.shuffle(values)
            values = values[:int(samples)]
            tmp = pd.DataFrame(values)
            tmp['label'] = i
            ss_df = pd.concat([ss_df, tmp])

        # calculate their silhouette scores
        ss_df['silhouette'] = silhouette_samples(ss_df.drop(columns='label').values, ss_df['label'].values, n_jobs=-1)
        ss_df['silhouette'] = ss_df['silhouette'].round(3)
        ss_df.columns = ss_df.columns.astype(str)
        write_table(ss_df, workdir, f'cluster_sscores_{kmeans.n_clusters}')

        # save the summary stats from this model
        summary['number'].append(n_clusters)
        summary['silhouette'].append(ss_df['silhouette'].mean())

    # save the summary stats
    write_table(pd.DataFrame(summary), workdir, 'cluster_sscores')
    return


def plot_clusters(workdir: str, x: np.ndarray = None, n_clusters: int or Iterable = 'all',
                  max_cols: int = 3, plt_width: int = 3, plt_height: int = 3, n_lines: int = 500) -> None:
    """
    Generate figures of the clustered FDC's

    Args:
        workdir: path to the project directory
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
        x = read_table(workdir, 'hindcast_fdc_trans').values

    size = x.shape[1]
    x_values = np.linspace(0, size, 5)
    x_ticks = np.linspace(0, 100, 5).astype(int)

    random_shuffler = np.random.default_rng()

    for model_file in _find_model_files(workdir, n_clusters):
        logger.info(f'Plotting {model_file}')

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
            dpi=500,
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

        fig.savefig(os.path.join(workdir, 'clusters', f'kmeans-clusters-{n_clusters}.png'))
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
    logger.info('Plotting silhouette scores')

    clusters_dir = os.path.join(workdir, 'clusters')

    for sscore_table in natsorted(glob.glob(os.path.join(clusters_dir, 'cluster_sscores_*.parquet'))):
        logger.info(f'Plotting {sscore_table}')
        n_clusters = int(sscore_table.split('_')[-1].split('.')[0])
        sscore_df = pd.read_parquet(sscore_table, engine='fastparquet')
        centers_df = read_table(workdir, f'cluster_centers_{n_clusters}')
        mean_ss = sscore_df['silhouette'].mean()

        # initialize the figure
        fig, (ax1, ax2) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(plt_width * 2 + 1, plt_height + 1),
            dpi=500,
            tight_layout=True,
        )

        # Plot 1 titles and labels
        ax1.set_title(f"Silhouette Plot (mean={mean_ss:.3f})")
        ax1.set_xlabel("Silhouette Score")
        ax1.set_ylabel("Cluster Label")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Plot 2 titles and labels
        ax2.set_title("Cluster Centers")
        ax2.set_xlabel("Exceedance Probability (%)")
        ax2.set_ylabel("Discharge Z-Score")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=mean_ss, color="red", linestyle="--")

        y_lower = 10
        for sub_cluster in range(int(n_clusters)):
            # select the rows applicable to the current sub cluster
            cluster_sscores = sscore_df[sscore_df['label'] == sub_cluster]['silhouette'].values.flatten()
            cluster_sscores.sort()

            n = cluster_sscores.shape[0]
            y_upper = y_lower + n

            color = cm.nipy_spectral(sub_cluster / int(n_clusters))
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_sscores,
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

        fig.savefig(os.path.join(clusters_dir, f'silhouette-diagram-{n_clusters}.png'))
        plt.close(fig)
    return
