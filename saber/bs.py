import logging
import os
import warnings
from multiprocessing import Pool

import contextily as cx
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_percentage_error

from ._metrics import kge2012
from ._metrics import nse
from .assign import _map_assign_ungauged
from .io import BTSTRP_METRIC_NAMES
from .io import COL_ASN_GID
from .io import COL_ASN_MID
from .io import COL_CID
from .io import COL_GID
from .io import COL_MID
from .io import COL_QMOD
from .io import COL_QOBS
from .io import COL_QSIM
from .io import COL_STRM_ORD
from .io import get_dir
from .io import get_state
from .io import read_gis
from .io import read_table
from .io import write_gis
from .io import write_table
from .saber import map_saber

__all__ = ['mp_table', 'metrics', 'mp_metrics', 'postprocess_metrics',
           'pie_charts', 'histograms_prepost', 'maps', 'boxplots_explanatory']

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

mpl.rcParams['font.family'] = 'serif'


def mp_table(assign_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates the assignment table for bootstrap validation by assigning each gauged stream to a different gauged stream
    following the same rules as all other gauges.

    Args:
        assign_df: pandas.DataFrame of the assignment table

    Returns:
        None
    """
    logger.info('Determining bootstrap assignments')

    # subset the assign dataframe to only rows which contain gauges - possible options to be assigned
    gauges_df = assign_df[assign_df[COL_GID].notna()].copy()

    with Pool(get_state('n_processes')) as p:
        bs_df = pd.concat(
            p.starmap(_map_mp_table, [[assign_df, gauges_df, row_idx] for row_idx in gauges_df.index])
        )

    write_table(bs_df, 'assign_table_bootstrap')
    return bs_df


def _map_mp_table(assign_df: pd.DataFrame, gauge_df: pd.DataFrame, row_idx: int) -> pd.DataFrame:
    """
    Helper function for mp_table which assigns a single row of the assignment table to a different gauged stream.
    Separate function so it can be pickled for multiprocessing.

    Args:
        assign_df: pandas.DataFrame of the assignment table
        gauge_df: pandas.DataFrame of the assignment table subset to only rows which contain gauges
        row_idx: the row number of the table to assign

    Returns:
        pandas.DataFrame of the row with the new assignment
    """
    return _map_assign_ungauged(assign_df, gauge_df.drop(row_idx), gauge_df.loc[row_idx][COL_MID])


def metrics(row_idx: int, assign_df: pd.DataFrame, gauge_data: str, hindcast_zarr: str) -> pd.DataFrame | None:
    """
    Performs bootstrap validation

    Args:
        row_idx: the row of the assignment table to remove and perform bootstrap validation with
        assign_df: pandas.DataFrame of the assignment table
        gauge_data: string path to the directory of observed data
        hindcast_zarr: string path to the hindcast streamflow dataset

    Returns:
        None
    """
    row = assign_df.loc[row_idx]

    try:
        corrected_df = map_saber(row[COL_MID], row[COL_ASN_MID], row[COL_ASN_GID], hindcast_zarr, gauge_data)

        if corrected_df is None:
            logger.warning(f'No corrected data for {row[COL_MID]}')
            return None
        if not (COL_QMOD in corrected_df.columns and COL_QSIM in corrected_df.columns):
            logger.warning(f'Missing adjusted and simulated columns')
            return None

        # create a dataframe of original and corrected streamflow that can be used for calculating metrics
        metrics_df = pd.read_csv(os.path.join(gauge_data, f'{row[COL_GID]}.csv'), index_col=0)
        metrics_df.columns = [COL_QOBS, ]
        metrics_df.index = pd.to_datetime(metrics_df.index)
        metrics_df = pd.merge(corrected_df, metrics_df, how='inner', left_index=True, right_index=True)

        # drop rows with inf or nan values
        metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan).dropna()

        # if the dataframe is empty (dates did not align or all rows were inf or NaN), return None
        if metrics_df.empty:
            logger.warning(f'Empty dataframe for {row[COL_MID]}')
            return None

        obs_values = metrics_df[COL_QOBS].values.flatten()
        sim_values = metrics_df[COL_QSIM].values.flatten()
        mod_values = np.squeeze(metrics_df[COL_QMOD].values.flatten())

        if mod_values.dtype == np.dtype('O'):
            mod_values = np.array(mod_values.tolist()).astype(np.float64).flatten()

        diff_sim = sim_values - obs_values
        diff_corr = mod_values - obs_values

        return pd.DataFrame({
            'me_sim': np.mean(diff_sim),
            'mae_sim': np.mean(np.abs(diff_sim)),
            'rmse_sim': np.sqrt(np.mean(diff_sim ** 2)),
            'r_sim': pearsonr(obs_values, sim_values).statistic,
            'mape_sim': mean_absolute_percentage_error(obs_values, sim_values),
            'nse_sim': nse(obs_values, sim_values),
            'kge_sim': kge2012(obs_values, sim_values),

            'me_corr': np.mean(diff_corr),
            'mae_corr': np.mean(np.abs(diff_corr)),
            'rmse_corr': np.sqrt(np.mean(diff_corr ** 2)),
            'r_corr': pearsonr(obs_values, mod_values).statistic,
            'mape_corr': mean_absolute_percentage_error(obs_values, mod_values),
            'nse_corr': nse(obs_values, mod_values),
            'kge_corr': kge2012(obs_values, mod_values),

            'reach_id': row[COL_MID],
            'gauge_id': row[COL_GID],
            'asgn_reach_id': row[COL_ASN_MID],
        }, index=[0, ])
    except Exception as e:
        logger.error(e)
        logger.error(f'Failed bootstrap validation for {row[COL_MID]}')
        return None


def mp_metrics(assign_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Performs bootstrap validation using multiprocessing.

    Args:
        assign_df: pandas.DataFrame of the assignment table

    Returns:
        None
    """
    logger.info('Collecting Performance Metrics')

    if assign_df is None:
        assign_df = read_table('assign_table_bootstrap')

    gauge_data_dir = get_state('gauge_data')
    hindcast_zarr = get_state('hindcast_zarr')

    # subset the assign dataframe to only rows which contain gauges & reset the index
    assign_df = assign_df[assign_df[COL_GID].notna()].reset_index(drop=True)

    with Pool(get_state('n_processes')) as p:
        metrics_df = pd.concat(
            p.starmap(
                metrics,
                [[idx, assign_df, gauge_data_dir, hindcast_zarr] for idx in assign_df.index]
            )
        )

    write_table(metrics_df, 'bootstrap_metrics')

    return metrics_df


def postprocess_metrics(bdf: pd.DataFrame = None,
                        gauge_gdf: gpd.GeoDataFrame = None,
                        bs_assign_table: pd.DataFrame = None) -> None:
    """
    Creates a geopackge of the gauge locations with added attributes for metrics calculated during the bootstrap
    validation.

    Args:
        bdf: pandas.DataFrame of the bootstrap metrics
        gauge_gdf: geopandas.GeoDataFrame of the gauge locations
        bs_assign_table: pandas.DataFrame of the assignment table generated for bootstrap validation

    Returns:
        None
    """
    if bdf is None:
        bdf = read_table('bootstrap_metrics')

    for metric in BTSTRP_METRIC_NAMES:
        # convert from string to float then prepare a column for the results.
        cols = [f'{metric}_sim', f'{metric}_corr']
        bdf[cols] = bdf[cols].astype(float)
        bdf[f'{metric}_diff'] = bdf[f'{metric}_corr'] - bdf[f'{metric}_sim']
        bdf[metric] = np.nan

    for metric in ['kge', 'nse', ]:
        # want to see increase or difference less than or equal to 0.2
        bdf.loc[bdf[f'{metric}_corr'] > bdf[f'{metric}_sim'], metric] = 2
        bdf.loc[bdf[f'{metric}_corr'] < bdf[f'{metric}_sim'], metric] = 0
        bdf.loc[np.abs(bdf[f'{metric}_corr'] - bdf[f'{metric}_sim']) <= 0.2, metric] = 1

    for metric in ['me', 'mae', 'rmse', 'mape', ]:
        # want to see decrease in absolute value or difference less than 10%
        bdf.loc[bdf[f'{metric}_corr'].abs() < bdf[f'{metric}_sim'].abs(), metric] = 2
        bdf.loc[bdf[f'{metric}_corr'].abs() > bdf[f'{metric}_sim'].abs(), metric] = 0
        bdf.loc[np.abs(bdf[f'{metric}_corr'] - bdf[f'{metric}_sim']) < bdf[
            f'{metric}_sim'].abs() * .1, metric] = 1

    for metric in ['r', ]:
        # want to see decrease in absolute value or difference less than 10%
        bdf.loc[bdf[f'{metric}_corr'].abs() > bdf[f'{metric}_sim'].abs(), metric] = 2
        bdf.loc[bdf[f'{metric}_corr'].abs() < bdf[f'{metric}_sim'].abs(), metric] = 0
        bdf.loc[np.abs(bdf[f'{metric}_corr'] - bdf[f'{metric}_sim']) < bdf[
            f'{metric}_sim'].abs() * .1, metric] = 1

    write_table(bdf, 'bootstrap_metrics')

    # make a geopackage of the gauges with the metrics
    if gauge_gdf is None:
        gauge_gdf = read_gis('gauge_gis')
    gauge_gdf = gauge_gdf.merge(bdf, on=COL_GID, how='left')
    write_gis(gauge_gdf, 'bootstrap_gauges')

    # add the metrics results to the assignment table
    if bs_assign_table is None:
        bs_assign_table = read_table('assign_table_bootstrap')
    bs_assign_table = bs_assign_table.merge(bdf, on=COL_GID, how='outer')
    write_table(bs_assign_table, 'assign_table_bootstrap')
    return


def pie_charts(bdf: pd.DataFrame = None) -> None:
    """
    Creates figures of the bootstrap metrics results

    Args:
        bdf: pandas.DataFrame of the bootstrap metrics

    Returns:
        None
    """
    if bdf is None:
        bdf = read_table('bootstrap_metrics')

    # make a grid of pie charts for each metric
    # determine the shape of the subplats that fit in 2 columns
    n_cols = 2
    n_rows = int(np.ceil(len(BTSTRP_METRIC_NAMES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 2 * n_rows), dpi=1500, tight_layout=True)
    fig.suptitle('Error Metric Changes Before and After Correction')
    for i, metric in enumerate(BTSTRP_METRIC_NAMES):
        ax = axes[i // n_cols, i % n_cols]
        ax.set_title(metric.upper())
        ax.pie(bdf[metric].value_counts().sort_index(),
               labels=['Worse', 'Same', 'Better'],
               autopct='%1.1f%%', )

    if len(BTSTRP_METRIC_NAMES) % n_cols != 0:
        axes[-1, -1].axis('off')
    fig.savefig(os.path.join(get_dir('validation'), 'figure_metric_change_pie.png'))
    plt.close(fig)
    return


def histograms_prepost(bdf: pd.DataFrame = None) -> None:
    """
    Creates histograms of the bootstrap metrics.

    Args:
        bdf: pandas.DataFrame of the bootstrap metrics

    Returns:
        None
    """
    if bdf is None:
        bdf = read_table('bootstrap_metrics')

    median_symbol = '$\widetilde{X}$'
    mean_symbol = '$\overline{X}$'
    std_symbol = '$\sigma$'

    for stat in BTSTRP_METRIC_NAMES:
        stat_df = bdf[[f'{stat}_corr', f'{stat}_sim']].astype(float).copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=2000, tight_layout=True, sharey=True)

        if stat == 'kge':
            binwidth = 0.1
            binrange = (-3, 1)
            ax1.axvline(-0.41, c='red', linestyle='--', label='KGE = -0.41')
            ax2.axvline(-0.41, c='red', linestyle='--', label='KGE = -0.41')
            units = None

        elif stat == 'me':
            binwidth = 5
            binrange = (-50, 50)
            units = 'm3/s'

        elif stat == 'mae':
            binwidth = 5
            binrange = (0, 120)
            units = 'm3/s'

        elif stat == 'rmse':
            binwidth = 5
            binrange = (0, 100)
            units = 'm3/s'

        elif stat == 'nse':
            binwidth = 0.1
            binrange = (-3, 1)
            units = None

        elif stat == 'r':
            binwidth = 0.05
            binrange = (-1, 1)
            units = None

        elif stat == 'mape':
            binwidth = 0.5
            binrange = (0, 15)
            units = '%'

        else:
            raise ValueError(f'Invalid statistic: {stat}')

        # remove extreme outliers caused by numerical issues
        for metric, threshold, replacement in (
                ('me', 10_000, 1_000),
                ('rmse', 1_000, 1_000),
                ('mae', 10_000, 1_000),
                ('mape', 50, 50),
                ('kge', 10, 3),
        ):
            if metric == stat:
                stat_df.loc[stat_df[f'{metric}_corr'] > threshold, f'{metric}_corr'] = replacement
                stat_df.loc[stat_df[f'{metric}_corr'] < -threshold, f'{metric}_corr'] = -replacement

        fig.suptitle(f'Gauge {stat.upper()} Histograms Before and After Correction')

        sim_med = stat_df[f'{stat}_sim'].median()
        sim_mean = stat_df[f'{stat}_sim'].mean()
        sim_std = stat_df[f'{stat}_sim'].std()
        corr_med = stat_df[f'{stat}_corr'].median()
        corr_mean = stat_df[f'{stat}_corr'].mean()
        corr_std = stat_df[f'{stat}_corr'].std()

        stat_df[stat_df <= binrange[0]] = binrange[0]
        stat_df[stat_df >= binrange[1]] = binrange[1]

        sns.histplot(stat_df, x=f'{stat}_sim', binwidth=binwidth, binrange=binrange, ax=ax1)
        sns.histplot(stat_df, x=f'{stat}_corr', binwidth=binwidth, binrange=binrange, ax=ax2)

        ax1.set_yticks(np.arange(0, ax1.get_yticks()[-1] + 1, 250), minor=True)

        ax1.set_xlim(binrange)
        ax2.set_xlim(binrange)
        ax1.grid(True, 'both', linestyle='--', alpha=0.4)
        ax2.grid(True, 'both', linestyle='--', alpha=0.4)

        ax1.set_ylabel('Number of Gauges')
        ax1.set_xlabel(f'Simulated {stat.upper()}' + (f' ({units})' if units else ''))
        ax2.set_xlabel(f'Corrected {stat.upper()}' + (f' ({units})' if units else ''))

        ax1.axvline(sim_med, c='green', label=f'{median_symbol}: {sim_med:.2f}')
        ax1.axvline(sim_mean, c='blue', label=f'{mean_symbol}: {sim_mean:.2f}')
        ax1.axvline(sim_std, c='black', alpha=0, label=f'{std_symbol}: {sim_std:.2f}')

        ax2.axvline(corr_med, c='green', label=f'{median_symbol}: {corr_med:.2f}')
        ax2.axvline(corr_mean, c='blue', label=f'{mean_symbol}: {corr_mean:.2f}')
        ax2.axvline(corr_std, c='black', alpha=0, label=f'{std_symbol}: {corr_std:.2f}')

        # make the labels visible
        ax1.legend()
        ax2.legend()

        fig.savefig(os.path.join(get_dir('validation'), f'figure_hist_prepost_{stat}.png'))
        plt.close(fig)
    return


def maps(bs_gdf: gpd.GeoDataFrame = None) -> None:
    """
    Creates maps of the bootstrap metrics.

    Args:
        bs_gdf: geopandas.GeoDataFrame of the gauge locations with bootstrap metrics

    Returns:
        None
    """
    if bs_gdf is None:
        bs_gdf = read_gis('bootstrap_gauges')
    bs_gdf = bs_gdf.to_crs(epsg=3857)

    # make a map of the ME, MAE, KGE, and RMSE
    for metric in ['me', 'mae', 'kge', 'rmse']:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8), dpi=1000, tight_layout=True)
        fig.suptitle(f'Gauge {metric.upper()}')

        # set the x and y limits
        ax1.set_xlim(-2e7, 2e7)
        ax2.set_xlim(-2e7, 2e7)
        ax3.set_xlim(-2e7, 2e7)

        ax1.set_ylim(-8.9e6, 1.19e7)
        ax2.set_ylim(-8.9e6, 1.19e7)
        ax3.set_ylim(-8.9e6, 1.19e7)

        bs_gdf[bs_gdf[metric] == 2].plot(column=metric, markersize=0.25, ax=ax1, color='green', zorder=1)
        bs_gdf[bs_gdf[metric] == 1].plot(column=metric, markersize=0.25, ax=ax2, color='blue', zorder=1)
        bs_gdf[bs_gdf[metric] == 0].plot(column=metric, markersize=0.25, ax=ax3, color='red', zorder=1)

        cx.add_basemap(ax=ax1, zoom=1, source=cx.providers.Esri.WorldGrayCanvas, attribution='', crs='EPSG:3857')
        cx.add_basemap(ax=ax2, zoom=1, source=cx.providers.Esri.WorldGrayCanvas, attribution='', crs='EPSG:3857')
        cx.add_basemap(ax=ax3, zoom=1, source=cx.providers.Esri.WorldGrayCanvas, attribution='', crs='EPSG:3857')

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])

        ax1.set_xlabel('')
        ax1.set_ylabel(f'Improved (n={bs_gdf[bs_gdf[metric] == 2].shape[0]})')
        ax2.set_xlabel('')
        ax2.set_ylabel(f'Equivalent (n={bs_gdf[bs_gdf[metric] == 1].shape[0]})')
        ax3.set_xlabel('')
        ax3.set_ylabel(f'Degraded (n={bs_gdf[bs_gdf[metric] == 0].shape[0]})')

        fig.savefig(os.path.join(get_dir('validation'), f'figure_map_{metric}.png'))
        plt.close(fig)
    return


def boxplots_explanatory(bdf: pd.DataFrame = None) -> None:
    """
    Creates boxplots of the error metrics before and after bootstrapping by explanatory variable

    Args:
        bdf: pandas.DataFrame of the bootstrap metrics

    Returns:
        None
    """
    if bdf is None:
        bdf = read_table('assign_table_bootstrap')

    bdf = bdf[bdf[COL_CID].notna()]
    bdf[COL_CID] = bdf[COL_CID].astype(float).astype(int)
    bdf[COL_STRM_ORD] = bdf[COL_STRM_ORD].astype(float).astype(int)

    for stat in BTSTRP_METRIC_NAMES:
        bdf[f'{stat}_sim'] = bdf[f'{stat}_sim'].astype(float)
        bdf[f'{stat}_corr'] = bdf[f'{stat}_corr'].astype(float)

    for exp_var, exp_name in [
        (COL_CID, 'Cluster'),
        (COL_STRM_ORD, 'Stream Order'),
    ]:
        stats_to_compare = ['me', 'r', 'kge', 'mape']
        fig, axes = plt.subplots(len(stats_to_compare), 2,
                                 figsize=(6, 2.25 * len(stats_to_compare)), dpi=1000, tight_layout=True)
        fig.suptitle(f' Error Metrics vs {exp_name} - Before and After Correction', fontweight='bold')
        axes[0][0].set_title('Simulated')
        axes[0][1].set_title('Bias Corrected')

        for idx, stat in enumerate(stats_to_compare):
            stat_label = stat.upper()
            if stat == 'kge':
                yrange = (-3, 1)
                axes[idx][0].axhline(-0.41, c='red', linestyle='--', label='KGE = -0.41')
                axes[idx][1].axhline(-0.41, c='red', linestyle='--', label='KGE = -0.41')
            elif stat == 'me':
                yrange = (-4000, 4000)
            elif stat == 'r':
                yrange = (-.25), 1
            elif stat == 'mape':
                yrange = (0, 5)
                stat_label = 'MAPE (as decimal)'
            else:
                raise ValueError(f'Invalid statistic: {stat}')

            sns.boxplot(data=bdf, x=exp_var, whis=(5, 95), showfliers=False, y=f'{stat}_sim', ax=axes[idx][0])
            sns.boxplot(data=bdf, x=exp_var, whis=(5, 95), showfliers=False, y=f'{stat}_corr', ax=axes[idx][1])

            axes[idx][0].set_ylim(yrange)
            axes[idx][1].set_ylim(yrange)

            axes[idx][0].set_ylabel(stat_label)
            axes[idx][1].set_ylabel('')

            axes[idx][0].yaxis.grid(True)
            axes[idx][1].yaxis.grid(True)
            axes[idx][1].set_yticklabels([])

            if idx == len(stats_to_compare) - 1:
                axes[idx][0].set_xlabel(exp_name)
                axes[idx][1].set_xlabel(exp_name)
            else:
                axes[idx][0].set_xticks([])
                axes[idx][1].set_xticks([])
                axes[idx][0].set_xlabel('')
                axes[idx][1].set_xlabel('')

        fig.savefig(os.path.join(get_dir('validation'), f'figure_boxplot_{exp_name.lower().replace(" ", "_")}.png'))
        plt.close(fig)

    return
