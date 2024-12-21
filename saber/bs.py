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
from dtaidistance import dtw
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

    # Load the observed data dataframe
    gauge_clstr = pd.read_csv('/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/dtw_iteration_2_minimum_distances_summary_with_climate.csv')
    # subset the assign dataframe to only rows which contain gauges - possible options to be assigned
    gauges_df = assign_df[assign_df[COL_GID].notna()].copy()
    # Calculate the minimum DTW distance and update the assign_df
    #gauges_df = calculate_dtw_min_distance(gauges_df, observed_data_df)
    gauges_df = gauges_df.merge(gauge_clstr[['File', 'clstr_gauge','Climate']], left_on='gauge_id', right_on='File', how='inner') #use this code to match gauge against cluster

    #note that I have only put climate column for gauge but when regionalization you might need to put Climate column for 7 million river

    assign_df = assign_df.merge(
        gauges_df[['gauge_id','clstr_gauge', 'Climate']],
        on='gauge_id',
        how='outer'
    )
    # with Pool(get_state('n_processes')) as p:
    #     bs_df = pd.concat(
    #         p.starmap(_map_cluster_gauge, [[main_curves_df, observed_row, gauge] for gauge in gauges_df['gauge_id']]))

    # subset the assign dataframe to only rows which contain gauges - possible options to be assigned
    with Pool(get_state('n_processes')) as p:
        bs_df = pd.concat(
            p.starmap(_map_mp_table, [[assign_df, gauges_df, row_idx] for row_idx in gauges_df.index])
        )

    write_table(bs_df, 'assign_table_bootstrap')
    return bs_df

# def calculate_min_dtw_index(main_curves_df: pd.DataFrame, observed_row: np.ndarray) -> str:
#     """
#     Calculate the DTW distance for each column in main_curves_df against the observed_row
#     and return the column index with the smallest distance.
#
#     Args:
#         main_curves_df: DataFrame containing the main curves (columns to compare against).
#         observed_row: A single row of observed data as a numpy array.
#
#     Returns:
#         The column index (or name) with the smallest DTW distance.
#     """
#     numpy_arrays = {col: main_curves_df[col].to_numpy() for col in main_curves_df.columns}
#     distances = {col: dtw.distance(numpy_arrays[col].flatten(), observed_row) for col in numpy_arrays}
#     min_distance_column = min(distances, key=distances.get)
#     return min_distance_column

# def _map_cluster_gauge(main_curves_df: pd.DataFrame, observed_row: pd.DataFrame, row_idx: int) -> pd.DataFrame:
#     """
#     Helper function for mp_table which assigns a single row of the assignment table to a different gauged stream.
#     Separate function so it can be pickled for multiprocessing.
#
#     Args:
#         assign_df: pandas.DataFrame of the assignment table
#         gauge_df: pandas.DataFrame of the assignment table subset to only rows which contain gauges
#         row_idx: the row number of the table to assign
#
#     Returns:
#         pandas.DataFrame of the row with the new assignment
#     """
#     return calculate_min_dtw_index(assign_df, gauge_df.drop(row_idx), gauge_df.loc[row_idx][COL_MID])


# def calculate_dtw_min_distance(gauges_df: pd.DataFrame, observed_data_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculates the minimum DTW distance for each row in the observed data against the columns in the main curves.
#     The result is saved as a new column in the assign_df DataFrame.
#
#     Args:
#         assign_df: pandas.DataFrame containing the assignment table, including 'gauge_id' column.
#         observed_data_df: pandas.DataFrame containing the observed data.
#
#     Returns:
#         assign_df: pandas.DataFrame updated with the minimum DTW distance information.
#     """
#     # Load the main curves dataframe
#     main_curves_df = pd.read_csv(
#         '/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/clusters/cluster_centers_5.csv')
#     main_curves_df = main_curves_df.sort_values(by='Unnamed: 0')
#     main_curves_df = main_curves_df.drop(columns=['Unnamed: 0'])
#
#     # Ensure both DataFrames are aligned on the 'gauge_id' and 'File' columns
#     gauges_df = gauges_df.merge(observed_data_df[['File']], left_on='gauge_id', right_on='File', how='inner')
#
#     # Initialize columns to store the minimum distance and corresponding column
#     gauges_df['min_distance_column'] = np.nan
#     gauges_df['min_distance_value'] = np.nan
#
#     # Iterate through each row in the observed_data_df
#     for index, row in observed_data_df.iterrows():
#         row_identifier = row['File']  # Assuming 'File' column is the identifier
#         observed_row = row[1:].to_numpy()  # Convert the rest of the row to a numpy array
#
#         # Calculate DTW distance for each column in main_curves_df
#         distances = {col: dtw.distance(main_curves_df[col].to_numpy().flatten(), observed_row) for col in
#                      main_curves_df.columns}
#
#         # Find the column with the minimum DTW distance for the current row
#         min_distance_column = min(distances, key=distances.get)
#         min_distance_value = distances[min_distance_column]
#
#         # Update the assign_df with the minimum distance and corresponding column
#         gauges_df.loc[gauges_df['File'] == row_identifier, 'min_distance_column'] = min_distance_column
#         gauges_df.loc[gauges_df['File'] == row_identifier, 'min_distance_value'] = min_distance_value
#
#     return gauges_df

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
       # metrics_df = pd.read_csv(os.path.join(gauge_data, f'{row[COL_GID]}.csv'), index_col=0,usecols=[0,2]).dropna(how='all')  ##use this to run anamoly csv file
        metrics_df.columns = [COL_QOBS, ]
        metrics_df.index = pd.to_datetime(metrics_df.index)
        metrics_df = pd.merge(corrected_df, metrics_df, how='inner', left_index=True, right_index=True)
        #save the bias corrected value
        #Bias_correct= '/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/Bias corrected Time series/1941 clstr fixed'
        #Save the metrics_df with the COL_MID in the file name
        #metrics_df.to_csv(os.path.join(Bias_correct, f'{row[COL_MID]}.csv'))

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
            'nrmse_corr': hs.nrmse_mean(mod_values, obs_values),
            'pearson_r_corr': hs.pearson_r(mod_values, obs_values),
            'r2_corr': hs.r_squared(mod_values, obs_values),
            'r2_sim': hs.r_squared(sim_values, obs_values),
            'nrmse_sim': hs.nrmse_mean(sim_values, obs_values),
            'pearson_r_sim': hs.pearson_r(sim_values, obs_values),
            'std_obs': np.std(obs_values),
            'std_sim':np.std(mod_values),
            'mean_obs':np.mean(obs_values),
            'mean_sim':np.mean(mod_values),
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

    #harek index of assigntable ko hercha ra tesko corresponding station khojcha in gauge directory ma and then on zarr
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

    for metric in ['kge', 'nse']: # A value of 2 shows improved performance. A value of 1 indicates small change (within 0.2 of simulated). A value of 0 signals declined performance.
        # want to see increase or difference less than or equal to 0.2
        bdf.loc[bdf[f'{metric}_corr'] > bdf[f'{metric}_sim'], metric] = 2
        bdf.loc[np.abs(bdf[f'{metric}_corr'] - bdf[f'{metric}_sim']) <= 0.2, metric] = 1
        bdf.loc[bdf[f'{metric}_corr'] < bdf[f'{metric}_sim'], metric] = 0

    for metric in ['me', 'mae', 'rmse']: #A 2 indicates reduced error. A '1' signals small error reduction (within 10% of simulated). A 0 shows increased error.
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
    labels_map = {0: 'Worse', 1: 'Same', 2: 'Better'}
    for i, metric in enumerate(['kge', 'me', 'mae', 'rmse']):
        ax = axes[i // 2, i % 2]
        ax.set_title(metric.upper())
        # Count occurrences of each category
        value_counts = bdf[metric].value_counts().sort_index()

        # Create labels dynamically based on available categories
        labels = [labels_map.get(category, category) for category in value_counts.index]

        # Plot pie chart with available categories
        ax.pie(value_counts,
               labels=labels,
               autopct='%1.1f%%')
    fig.savefig(os.path.join(get_dir('validation'), 'figure_metric_change_pie.png'))

    return
