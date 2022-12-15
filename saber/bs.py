import logging
import os
import warnings
from multiprocessing import Pool

import geopandas as gpd
import hydrostats as hs
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .assign import _map_assign_ungauged
from .io import COL_ASN_GID
from .io import COL_ASN_MID
from .io import COL_GID
from .io import COL_MID
from .io import COL_QMOD
from .io import COL_QOBS
from .io import COL_QSIM
from .io import get_dir
from .io import get_state
from .io import read_gis
from .io import read_table
from .io import write_gis
from .io import write_table
from .saber import map_saber

__all__ = ['mp_table', 'metrics', 'mp_metrics', 'histograms', 'postprocess_metrics', 'pie_charts']

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


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
            'nse_sim': hs.nse(sim_values, obs_values),
            'kge_sim': hs.kge_2012(sim_values, obs_values),

            'me_corr': np.mean(diff_corr),
            'mae_corr': np.mean(np.abs(diff_corr)),
            'rmse_corr': np.sqrt(np.mean(diff_corr ** 2)),
            'nse_corr': hs.nse(mod_values, obs_values),
            'kge_corr': hs.kge_2012(mod_values, sim_values),

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


def postprocess_metrics(bdf: pd.DataFrame = pd.DataFrame or None, gauge_gdf: gpd.GeoDataFrame = None) -> None:
    """
    Creates a geopackge of the gauge locations with added attributes for metrics calculated during the bootstrap
    validation.

    Args:
        bdf: pandas.DataFrame of the bootstrap metrics
        gauge_gdf: geopandas.GeoDataFrame of the gauge locations

    Returns:
        None
    """
    if bdf is None:
        bdf = read_table('bootstrap_metrics')

    for metric in ['me', 'mae', 'rmse', 'kge', 'nse']:
        # convert from string to float then prepare a column for the results.
        cols = [f'{metric}_sim', f'{metric}_corr']
        bdf[cols] = bdf[cols].astype(float)
        bdf[metric] = np.nan

    for metric in ['kge', 'nse']:
        # want to see increase or difference less than or equal to 0.2
        bdf.loc[bdf[f'{metric}_corr'] > bdf[f'{metric}_sim'], metric] = 2
        bdf.loc[np.abs(bdf[f'{metric}_corr'] - bdf[f'{metric}_sim']) <= 0.2, metric] = 1
        bdf.loc[bdf[f'{metric}_corr'] < bdf[f'{metric}_sim'], metric] = 0

    for metric in ['me', 'mae', 'rmse']:
        # want to see decrease in absolute value or difference less than 10%
        bdf.loc[bdf[f'{metric}_corr'].abs() < bdf[f'{metric}_sim'].abs(), metric] = 2
        bdf.loc[np.abs(bdf[f'{metric}_corr'] - bdf[f'{metric}_sim']) < bdf[
            f'{metric}_sim'].abs() * .1, metric] = 1
        bdf.loc[bdf[f'{metric}_corr'].abs() > bdf[f'{metric}_sim'].abs(), metric] = 0

    write_table(bdf, 'bootstrap_metrics')

    if gauge_gdf is None:
        gauge_gdf = read_gis('gauge_gis')
    gauge_gdf = gauge_gdf.merge(bdf, on=COL_GID, how='left')
    write_gis(gauge_gdf, 'bootstrap_gauges')
    return


def histograms(bdf: pd.DataFrame = None) -> None:
    """
    Creates histograms of the bootstrap metrics.

    Args:
        bdf: pandas.DataFrame of the bootstrap metrics

    Returns:
        None
    """
    if bdf is None:
        bdf = read_table('bootstrap_metrics')

    for stat in ['me', 'mae', 'rmse', 'nse', 'kge']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=2000, tight_layout=True, sharey=True)

        if stat == 'kge':
            binwidth = 0.25
            binrange = (-6, 1)
            ax1.axvline(-0.44, c='red', linestyle='--')
            ax2.axvline(-0.44, c='red', linestyle='--')

        elif stat == 'nse':
            binwidth = 0.25
            binrange = (-6, 1)

        elif stat == 'me':
            binwidth = 20
            binrange = (-200, 200)

        elif stat == 'mae':
            binwidth = 30
            binrange = (0, 300)

        elif stat == 'rmse':
            binwidth = 20
            binrange = (0, 200)

        else:
            raise ValueError(f'Invalid statistic: {stat}')

        fig.suptitle(f'Bootstrap Validation: {stat.upper()}')
        ax1.grid(True, 'both', zorder=0, linestyle='--')
        ax2.grid(True, 'both', zorder=0, linestyle='--')
        ax1.set_xlim(binrange)
        ax2.set_xlim(binrange)

        stat_df = bdf[[f'{stat}_corr', f'{stat}_sim']].astype(float).copy()
        stat_df[stat_df <= binrange[0]] = binrange[0]
        stat_df[stat_df >= binrange[1]] = binrange[1]

        sns.histplot(stat_df, x=f'{stat}_sim', binwidth=binwidth, binrange=binrange, ax=ax1)
        sns.histplot(stat_df, x=f'{stat}_corr', binwidth=binwidth, binrange=binrange, ax=ax2)

        ax1.axvline(stat_df[f'{stat}_sim'].median(), c='green')
        ax2.axvline(stat_df[f'{stat}_corr'].median(), c='green')

        fig.savefig(os.path.join(get_dir('validation'), f'figure_bootstrap_{stat}.png'))
        plt.close(fig)
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
    fig, axes = plt.subplots(2, 2, figsize=(4, 4), dpi=2000, tight_layout=True)
    fig.suptitle('Bootstrap Validation Metrics')
    for i, metric in enumerate(['kge', 'me', 'mae', 'rmse']):
        ax = axes[i // 2, i % 2]
        ax.set_title(metric.upper())
        ax.pie(bdf[metric].value_counts().sort_index(),
               labels=['Worse', 'Same', 'Better'],
               autopct='%1.1f%%')
    fig.savefig(os.path.join(get_dir('validation'), 'figure_metric_change_pie.png'))

    return
