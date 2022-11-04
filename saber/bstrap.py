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

from .assign import map_assign_ungauged
from .calibrate import map_saber
from .io import asgn_gid_col
from .io import asgn_mid_col
from .io import dir_valid
from .io import gid_col
from .io import mid_col
from .io import q_mod
from .io import q_obs
from .io import q_sim
from .io import write_table

__all__ = ['mp_table', 'metrics', 'mp_metrics', 'plots', 'merge_metrics_and_gis']

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def mp_table(workdir: str, assign_df: pd.DataFrame, n_processes: int = 1) -> pd.DataFrame:
    """
    Generates the assignment table for bootstrap validation by assigning each gauged stream to a different gauged stream
    following the same rules as all other gauges.

    Args:
        workdir: the project working directory path
        assign_df: pandas.DataFrame of the assignment table
        n_processes: number of processes to use for multiprocessing, passed to Pool

    Returns:
        None
    """
    # subset the assign dataframe to only rows which contain gauges & reset the index
    assign_df = assign_df[assign_df[gid_col].notna()].reset_index(drop=True)

    with Pool(n_processes) as p:
        bootstrap_assign_df = pd.concat(
            p.starmap(_map_mp_table, [[assign_df, row_idx] for row_idx in assign_df.index])
        )

    write_table(bootstrap_assign_df, workdir, 'assign_table_bootstrap')
    return bootstrap_assign_df


def _map_mp_table(assign_df: pd.DataFrame, row_idx: int) -> pd.DataFrame:
    """
    Helper function for mp_table which assigns a single row of the assignment table to a different gauged stream.
    Separate function so it can be pickled for multiprocessing.

    Args:
        assign_df: pandas.DataFrame of the assignment table
        row_idx: the row number of the table to assign

    Returns:
        pandas.DataFrame of the row with the new assignment
    """
    return map_assign_ungauged(assign_df, assign_df.drop(row_idx), assign_df.loc[row_idx][mid_col])


def metrics(row_idx: int, assign_df: pd.DataFrame, gauge_data: str, hds: str) -> pd.DataFrame | None:
    """
    Performs bootstrap validation.

    Args:
        row_idx: the row of the assignment table to remove and perform bootstrap validation with
        assign_df: pandas.DataFrame of the assignment table
        gauge_data: string path to the directory of observed data
        hds: string path to the hindcast streamflow dataset

    Returns:
        None
    """
    try:
        row = assign_df.loc[row_idx]
        corrected_df = map_saber(
            row[mid_col],
            row[asgn_mid_col],
            row[asgn_gid_col],
            hds,
            gauge_data,
        )

        if corrected_df is None:
            logger.warning(f'No corrected data for {row[mid_col]}')
            return None
        if not (q_mod in corrected_df.columns and q_sim in corrected_df.columns):
            logger.warning(f'Missing adjusted and simulated columns')
            return None

        # create a dataframe of original and corrected streamflow that can be used for calculating metrics
        metrics_df = pd.read_csv(os.path.join(gauge_data, f'{row[gid_col]}.csv'), index_col=0)
        metrics_df.columns = [q_obs, ]
        metrics_df.index = pd.to_datetime(metrics_df.index)
        metrics_df = pd.merge(corrected_df, metrics_df, how='inner', left_index=True, right_index=True)

        # drop rows with inf or nan values
        metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan).dropna()

        # if the dataframe is empty (dates did not align or all rows were inf or NaN), return None
        if metrics_df.empty:
            logger.warning(f'Empty dataframe for {row[mid_col]}')
            return None

        obs_values = metrics_df[q_obs].values.flatten()
        sim_values = metrics_df[q_sim].values.flatten()
        mod_values = np.squeeze(metrics_df[q_mod].values.flatten())

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

            'reach_id': row[mid_col],
            'gauge_id': row[gid_col],
            'asgn_reach_id': row[asgn_mid_col],
        }, index=[0, ])
    except Exception as e:
        logger.error(e)
        logger.error(f'Failed bootstrap validation for {row[mid_col]}')
        return None


def mp_metrics(workdir: str, assign_df: pd.DataFrame, gauge_data: str, hds: str, n_processes: int = 1) -> pd.DataFrame:
    """
    Performs bootstrap validation using multiprocessing.

    Args:
        workdir: the project working directory
        assign_df: pandas.DataFrame of the assignment table
        gauge_data: string path to the directory of observed data
        hds: string path to the hindcast streamflow dataset
        n_processes: number of processes to use for multiprocessing, passed to Pool

    Returns:
        None
    """
    # subset the assign dataframe to only rows which contain gauges & reset the index
    assign_df = assign_df[assign_df[gid_col].notna()].reset_index(drop=True)

    with Pool(n_processes) as p:
        metrics_df = pd.concat(
            p.starmap(
                metrics,
                [[idx, assign_df, gauge_data, hds] for idx in assign_df.index]
            )
        )

    write_table(metrics_df, workdir, 'bootstrap_metrics')

    return metrics_df


def plots(workdir: str, bdf: pd.DataFrame = None) -> None:
    """

    Args:
        workdir: the project working directory
        bdf: pandas.DataFrame of the bootstrap metrics

    Returns:
        None
    """
    if bdf is None:
        bdf = pd.read_csv(os.path.join(workdir, dir_valid, 'bootstrap_metrics.csv'))

    for stat in ['me', 'mae', 'rmse', 'nse', 'kge']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=1500, tight_layout=True, sharey=True)

        if stat == 'kge':
            binwidth = 0.5
            binrange = (-6, 1)
            ax1.axvline(-0.44, c='red', linestyle='--')
            ax2.axvline(-0.44, c='red', linestyle='--')

        elif stat == 'nse':
            binwidth = 0.5
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

        stat_df = bdf[[f'{stat}_corr', f'{stat}_sim']].reset_index(drop=True).copy()
        stat_df[stat_df <= binrange[0]] = binrange[0]
        stat_df[stat_df >= binrange[1]] = binrange[1]

        sns.histplot(stat_df, x=f'{stat}_sim', binwidth=binwidth, binrange=binrange, ax=ax1)
        sns.histplot(stat_df, x=f'{stat}_corr', binwidth=binwidth, binrange=binrange, ax=ax2)

        ax1.axvline(stat_df[f'{stat}_sim'].median(), c='green')
        ax2.axvline(stat_df[f'{stat}_corr'].median(), c='green')

        fig.savefig(os.path.join(workdir, dir_valid, f'bootstrap_{stat}.png'))
    return


def merge_metrics_and_gis(workdir: str, gdf: gpd.GeoDataFrame or str, bdf: pd.DataFrame = pd.DataFrame or None) -> None:
    """
    Creates a matplolib map of KGE metrics

    Args:
        workdir: the project working directory
        gdf: geopandas.GeoDataFrame of the model reaches
        bdf: pandas.DataFrame of the bootstrap metrics

    Returns:
        None
    """
    if isinstance(gdf, str):
        gdf = gpd.read_file(gdf)

    if bdf is None:
        bdf = pd.read_csv(os.path.join(workdir, dir_valid, 'bootstrap_metrics.csv'))

    gdf = gdf.merge(bdf, on=gid_col, how='left')

    for metric in ['me', 'mae', 'rmse', 'kge', 'nse']:
        # convert from string to float then prepare a column for the results.
        cols = [f'{metric}_sim', f'{metric}_corr']
        gdf[cols] = gdf[cols].astype(float)
        gdf[metric] = np.nan

    for metric in ['kge', 'nse']:
        # want to see increase or difference less than or equal to 0.2
        gdf.loc[gdf[f'{metric}_corr'] > gdf[f'{metric}_sim'], metric] = 2
        gdf.loc[np.abs(gdf[f'{metric}_corr'] - gdf[f'{metric}_sim']) <= 0.2, metric] = 1
        gdf.loc[gdf[f'{metric}_corr'] < gdf[f'{metric}_sim'], metric] = 0

    for metric in ['me', 'mae', 'rmse']:
        # want to see decrease in absolute value or difference less than 10%
        gdf.loc[gdf[f'{metric}_corr'].abs() < gdf[f'{metric}_sim'].abs(), metric] = 2
        gdf.loc[np.abs(gdf[f'{metric}_corr'] - gdf[f'{metric}_sim']) < gdf[f'{metric}_sim'].abs() * .1, metric] = 1
        gdf.loc[gdf[f'{metric}_corr'].abs() > gdf[f'{metric}_sim'].abs(), metric] = 0

    gdf.to_file(os.path.join(workdir, dir_valid, 'bootstrap_metrics_all.gpkg'), driver='GPKG')
    return
