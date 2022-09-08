import logging
import datetime
import os
import statistics

import hydrostats as hs
import hydrostats.data as hd
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy import interpolate, stats

from .io import asgn_gid_col
from .io import asgn_mid_col
from .io import cal_nc_name
from .io import metric_list
from .io import metric_nc_name_list
from .io import mid_col
from .io import table_hindcast

logger = logging.getLogger(__name__)


def calibrate(sim_flow_a: pd.DataFrame, obs_flow_a: pd.DataFrame, sim_flow_b: pd.DataFrame = None,
              fix_seasonally: bool = True, empty_months: str = 'skip',
              drop_outliers: bool = False, outlier_threshold: int or float = 2.5,
              filter_scalar_fdc: bool = False, filter_range: tuple = (0, 80),
              extrapolate: str = 'nearest', fill_value: int or float = None,
              fit_gumbel: bool = False, fit_range: tuple = (10, 90), ) -> pd.DataFrame:
    """
    Removes the bias from simulated discharge using the SABER method.

    Given simulated and observed discharge at location A, removes bias from simulated data at point A.
    Given simulated and observed discharge at location A, removes bias from simulated data at point B, if given B.

    Args:
        sim_flow_a (pd.DataFrame): simulated hydrograph at point A. should contain a datetime index with daily values
            and a single column of discharge values.
        obs_flow_a (pd.DataFrame): observed hydrograph at point A. should contain a datetime index with daily values
            and a single column of discharge values.
        sim_flow_b (pd.DataFrame): (optional) simulated hydrograph at point B to correct using scalar flow duration
            curve mapping and the bias relationship at point A. should contain a datetime index with daily values
            and a single column of discharge values.

        fix_seasonally (bool): fix on a monthly (True) or annual (False) basis
        empty_months (str): how to handle months in the simulated data where no observed data are available. Options:
            "skip": ignore simulated data for months without

        drop_outliers (bool): flag to exclude outliers
        outlier_threshold (int or float): number of std deviations from mean to exclude from flow duration curve

        filter_scalar_fdc (bool): flag to filter the scalar flow duration curve
        filter_range (tuple): lower and upper bounds of the filter range

        extrapolate (str): method to use for extrapolation. Options: nearest, const, linear, average, max, min
        fill_value (int or float): value to use for extrapolation when extrapolate_method='const'

        fit_gumbel (bool): flag to replace extremely low/high corrected flows with values from Gumbel type 1
        fit_range (tuple): lower and upper bounds of exceedance probabilities to replace with Gumbel values

    Returns:
        pd.DataFrame with a DateTime index and columns with corrected flow, uncorrected flow, the scalar adjustment
        factor applied to correct the discharge, and the percentile of the uncorrected flow (in the seasonal grouping,
        if applicable).
    """
    if sim_flow_b is None:
        sim_flow_b = sim_flow_a.copy()
    if fix_seasonally:
        # list of the unique months in the historical simulation. should always be 1->12 but just in case...
        monthly_results = []
        for month in sorted(set(sim_flow_a.index.strftime('%m'))):
            # filter data to current iteration's month
            mon_obs_data = obs_flow_a[obs_flow_a.index.month == int(month)].dropna()

            if mon_obs_data.empty:
                if empty_months == 'skip':
                    continue
                else:
                    raise ValueError(f'Invalid value for argument "empty_months". Given: {empty_months}.')

            mon_sim_data = sim_flow_a[sim_flow_a.index.month == int(month)].dropna()
            mon_cor_data = sim_flow_b[sim_flow_b.index.month == int(month)].dropna()
            monthly_results.append(calibrate(
                mon_sim_data, mon_obs_data, mon_cor_data,
                fix_seasonally=False, empty_months=empty_months,
                drop_outliers=drop_outliers, outlier_threshold=outlier_threshold,
                filter_scalar_fdc=filter_scalar_fdc, filter_range=filter_range,
                extrapolate=extrapolate, fill_value=fill_value,
                fit_gumbel=fit_gumbel, fit_range=fit_range, )
            )
        # combine the results from each monthly into a single dataframe (sorted chronologically) and return it
        return pd.concat(monthly_results).sort_index()

    # compute the flow duration curves
    if drop_outliers:
        sim_fdc_a = calc_fdc(_drop_outliers_by_zscore(sim_flow_a, threshold=outlier_threshold), col_name='Q_sim')
        sim_fdc_b = calc_fdc(_drop_outliers_by_zscore(sim_flow_b, threshold=outlier_threshold), col_name='Q_sim')
        obs_fdc = calc_fdc(_drop_outliers_by_zscore(obs_flow_a, threshold=outlier_threshold), col_name='Q_obs')
    else:
        sim_fdc_a = calc_fdc(sim_flow_a, col_name='Q_sim')
        sim_fdc_b = calc_fdc(sim_flow_b, col_name='Q_sim')
        obs_fdc = calc_fdc(obs_flow_a, col_name='Q_obs')

    # calculate the scalar flow duration curve (at point A with simulated and observed data)
    scalar_fdc = calc_sfdc(sim_fdc_a['flow'].values.flatten(), obs_fdc['flow'].values.flatten())
    if filter_scalar_fdc:
        scalar_fdc = scalar_fdc[scalar_fdc['p_exceed'].between(filter_range[0], filter_range[1])]

    # make interpolators: Q_b -> p_exceed, p_exceed -> scalars_a
    # flow at B converted to exceedance probabilities, then matched with the scalar computed at point A
    flow_to_percent = _make_interpolator(sim_fdc_b.values,
                                         sim_fdc_b.index,
                                         extrap=extrapolate,
                                         fill_value=fill_value)

    percent_to_scalar = _make_interpolator(scalar_fdc.index,
                                           scalar_fdc.values,
                                           extrap=extrapolate,
                                           fill_value=fill_value)

    # apply interpolators to correct flows at B with data from A
    qb_original = sim_flow_b.values.flatten()
    p_exceed = flow_to_percent(qb_original)
    scalars = percent_to_scalar(p_exceed)
    qb_adjusted = qb_original / scalars

    if fit_gumbel:
        qb_adjusted = _fit_extreme_values_to_gumbel(qb_adjusted, p_exceed, fit_range)

    return pd.DataFrame(data=np.transpose([qb_adjusted, qb_original, scalars, p_exceed]),
                        index=sim_flow_b.index.to_list(),
                        columns=('Q_adjusted', 'Q_original', 'scalars', 'p_exceed'))


def calc_fdc(flows: np.array, steps: int = 201, col_name: str = 'Q') -> pd.DataFrame:
    """
    Compute flow duration curve (exceedance probabilities) from a list of flows

    Args:
        flows: array of flows
        steps: number of steps (exceedance probabilities) to use in the FDC
        col_name: name of the column in the returned dataframe

    Returns:
        pd.DataFrame with index 'p_exceed' and columns 'Q' (or col_name)
    """
    # calculate the FDC and save to parquet
    exceed_prob = np.linspace(100, 0, steps)
    fdc_flows = np.nanpercentile(flows, exceed_prob)
    df = pd.DataFrame(fdc_flows, columns=[col_name, ], index=exceed_prob)
    df.index.name = 'p_exceed'
    return df


def calc_sfdc(sim_fdc: pd.DataFrame, obs_fdc: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the scalar flow duration curve (exceedance probabilities) from two flow duration curves

    Args:
        sim_fdc: simulated flow duration curve
        obs_fdc: observed flow duration curve

    Returns:
        pd.DataFrame with index 'p_exceed' and columns 'Q'
    """
    scalars_df = pd.DataFrame(np.divide(sim_fdc.values, obs_fdc.values), columns=['scalars'], index=sim_fdc.index)
    scalars_df.replace(np.inf, np.nan, inplace=True)
    scalars_df.dropna(inplace=True)
    return scalars_df


def _drop_outliers_by_zscore(df: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
    """
    Drop outliers from a dataframe by their z-score and a threshold
    Based on https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
    Args:
        df: dataframe to drop outliers from
        threshold: z-score threshold

    Returns:
        pd.DataFrame with outliers removed
    """
    return df[(np.abs(stats.zscore(df)) < threshold).all(axis=1)]


def _filter_sfdc(sfdc: pd.DataFrame, filter_range: list) -> pd.DataFrame:
    """
    Filter the scalar flow duration curve by the specified range

    Args:
        sfdc: scalar flow duration curve DataFrame
        filter_range: list of [lower_bound: int or float, upper_bound: int or float]

    Returns:
        pd.DataFrame: filtered scalar flow duration curve
    """
    return sfdc[np.logical_and(sfdc.index > filter_range[0], sfdc.index < filter_range[1])]


def _make_interpolator(x: np.array, y: np.array, extrap: str = 'nearest',
                       fill_value: int or float = None) -> interpolate.interp1d:
    """
    Make an interpolator from two arrays

    Args:
        x: x values
        y: y values
        extrap: method for extrapolation: nearest, const, linear, average, max, min
        fill_value: value to use when extrap='const'

    Returns:
        interpolate.interp1d
    """
    # make interpolator which converts the percentiles to scalars
    if extrap == 'nearest':
        return interpolate.interp1d(x, y, fill_value='extrapolate', kind='nearest')
    elif extrap == 'const':
        if fill_value is None:
            raise ValueError('Must provide the const kwarg when extrap_method="const"')
        return interpolate.interp1d(x, y, fill_value=fill_value, bounds_error=False)
    elif extrap == 'linear':
        return interpolate.interp1d(x, y, fill_value='extrapolate')
    elif extrap == 'average':
        return interpolate.interp1d(x, y, fill_value=np.mean(y), bounds_error=False)
    elif extrap == 'max' or extrap == 'maximum':
        return interpolate.interp1d(x, y, fill_value=np.max(y), bounds_error=False)
    elif extrap == 'min' or extrap == 'minimum':
        return interpolate.interp1d(x, y, fill_value=np.min(y), bounds_error=False)
    else:
        raise ValueError('Invalid extrapolation method provided')


def _solve_gumbel1(std, xbar, rp):
    """
    Solves the Gumbel Type I pdf = exp(-exp(-b))

    Args:
        std: standard deviation of the dataset
        xbar: mean of the dataset
        rp: return period to calculate in years

    Returns:
        float: discharge value
    """
    return -np.log(-np.log(1 - (1 / rp))) * std * .7797 + xbar - (.45 * std)


def _fit_extreme_values_to_gumbel(q_adjust: np.array, p_exceed: np.array, fit_range: tuple = None) -> np.array:
    """
    Replace the extreme values from the corrected data with values based on the gumbel distribution

    Args:
        q_adjust: adjusted flows to be refined
        p_exceed: exceedance probabilities of the adjusted flows
        fit_range: range of exceedance probabilities to fit to the Gumbel distribution

    Returns:
        np.array of the flows with the extreme values replaced
    """
    all_values = pd.DataFrame(np.transpose([q_adjust, p_exceed]), columns=('q', 'p'))
    # compute the average and standard deviation for the values within the user specified fit_range
    mid_vals = all_values.copy()
    mid_vals = mid_vals[np.logical_and(mid_vals['p'] > fit_range[0], mid_vals['p'] < fit_range[1])]
    xbar = statistics.mean(mid_vals['q'].values)
    std = statistics.stdev(mid_vals['q'].values, xbar)

    # todo check that this is correct
    q = []
    for p in mid_vals[mid_vals['p'] <= fit_range[0]]['p'].tolist():
        q.append(_solve_gumbel1(std, xbar, 1 / (1 - (p / 100))))
    mid_vals.loc[mid_vals['p'] <= fit_range[0], 'q'] = q

    q = []
    for p in mid_vals[mid_vals['p'] >= fit_range[1]]['p'].tolist():
        if p >= 100:
            p = 99.999
        q.append(_solve_gumbel1(std, xbar, 1 / (1 - (p / 100))))
    mid_vals.loc[mid_vals['p'] >= fit_range[1], 'q'] = q

    qb_adjusted = mid_vals['q'].values
    return qb_adjusted


def calibrate_region(workdir: str, assign_table: pd.DataFrame,
                     gauge_table: pd.DataFrame = None, obs_data_dir: str = None) -> None:
    """
    Creates a netCDF of all corrected flows in a region.

    Args:
        workdir: path to project working directory
        assign_table: the assign_table dataframe
        gauge_table: path to the gauge table
        obs_data_dir: path to the observed data

    Returns:
        None
    """
    # todo create a parquet instead of a netcdf
    if gauge_table is None:
        gauge_table = pd.read_csv(os.path.join(workdir, 'gis_inputs', 'gauge_table.csv'))
    if obs_data_dir is None:
        obs_data_dir = os.path.join(workdir, 'data_inputs', 'obs_csvs')

    bcs_nc_path = os.path.join(workdir, cal_nc_name)
    ts = pd.read_pickle(os.path.join(workdir, 'data_processed', table_hindcast))

    # create the new netcdf
    bcs_nc = nc.Dataset(bcs_nc_path, 'w')

    # set up the dimensions
    t_size = ts.values.shape[0]
    m_size = ts.values.shape[1]
    c_size = 2
    bcs_nc.createDimension('time', t_size)
    bcs_nc.createDimension('model_id', m_size)
    bcs_nc.createDimension('corrected', c_size)

    # coordinate variables
    bcs_nc.createVariable('time', 'i', ('time',), zlib=True, shuffle=True, fill_value=-9999)
    bcs_nc.createVariable('model_id', 'i', ('model_id',), zlib=True, shuffle=True, fill_value=-9999)
    bcs_nc.createVariable('corrected', 'i', ('corrected',), zlib=True, shuffle=True, fill_value=-1)

    # other variables
    bcs_nc.createVariable('flow_sim', 'f4', ('time', 'model_id'), zlib=True, shuffle=True, fill_value=np.nan)
    bcs_nc.createVariable('flow_bc', 'f4', ('time', 'model_id'), zlib=True, shuffle=True, fill_value=np.nan)
    bcs_nc.createVariable('percentiles', 'f4', ('time', 'model_id'), zlib=True, shuffle=True, fill_value=np.nan)
    bcs_nc.createVariable('scalars', 'f4', ('time', 'model_id'), zlib=True, shuffle=True, fill_value=np.nan)
    for metric in metric_nc_name_list:
        bcs_nc.createVariable(metric, 'f4', ('model_id', 'corrected'), zlib=True, shuffle=True, fill_value=np.nan)

    # times from the datetime index
    times = ts.index.values.astype(datetime.datetime)
    # convert nanoseconds to milliseconds
    times = times / 1000000
    # convert to dates
    times = nc.num2date(times, 'milliseconds since 1970-01-01 00:00:00', calendar='standard')
    # convert to a simpler unit
    times = nc.date2num(times, 'days since 1970-01-01 00:00:00', calendar='standard')

    # fill the values we already know
    bcs_nc['time'].unit = 'days since 1970-01-01 00:00:00+00'
    bcs_nc['time'][:] = times
    bcs_nc['model_id'][:] = ts.columns.to_list()
    bcs_nc['corrected'][:] = np.array((0, 1))
    bcs_nc['flow_sim'][:] = ts.values

    bcs_nc.sync()

    # set up arrays to compute the corrected, percentile and scalar arrays
    c_array = np.array([np.nan] * t_size * m_size).reshape((t_size, m_size))
    p_array = np.array([np.nan] * t_size * m_size).reshape((t_size, m_size))
    s_array = np.array([np.nan] * t_size * m_size).reshape((t_size, m_size))

    computed_metrics = {}
    for metric in metric_list:
        computed_metrics[metric] = np.array([np.nan] * m_size * c_size).reshape(m_size, c_size)

    errors = {'g1': 0, 'g2': 0, 'g3': 0, 'g4': 0}
    for idx, triple in enumerate(assign_table[[mid_col, asgn_mid_col, asgn_gid_col]].values):
        try:
            if idx % 25 == 0:
                logger.info(f'\n\t\t{idx + 1}/{m_size}')

            model_id, asgn_mid, asgn_gid = triple
            if np.isnan(asgn_gid) or np.isnan(asgn_mid):
                continue
            model_id = int(model_id)
            asgn_mid = int(asgn_mid)
            asgn_gid = int(asgn_gid)
        except Exception as e:
            errors['g1'] += 1
            continue

        try:
            obs_df = pd.read_csv(os.path.join(obs_data_dir, f'{asgn_gid}.csv'), index_col=0, parse_dates=True)
            obs_df = obs_df.dropna()
        except Exception as e:
            logger.info('failed to read the observed data')
            logger.info(e)
            errors['g2'] += 1
            continue

        try:
            calibrated_df = calibrate(ts[[asgn_mid]], obs_df, ts[[model_id]], )
            c_array[:, idx] = calibrated_df['flow'].values
            p_array[:, idx] = calibrated_df['percentile'].values
            s_array[:, idx] = calibrated_df['scalars'].values
        except Exception as e:
            logger.info('failed during the calibration step')
            logger.info(e)
            errors['g3'] += 1
            continue

        try:
            if model_id in gauge_table['model_id'].values:
                correct_id = gauge_table.loc[gauge_table['model_id'] == model_id, 'gauge_id'].values[0]
                obs_df = pd.read_csv(os.path.join(obs_data_dir, f'{correct_id}.csv'), index_col=0, parse_dates=True)
                sim_obs_stats = hs.make_table(hd.merge_data(sim_df=ts[[model_id]], obs_df=obs_df), metric_list)
                bcs_obs_stats = hs.make_table(hd.merge_data(sim_df=calibrated_df[['flow']], obs_df=obs_df), metric_list)
                for metric in metric_list:
                    computed_metrics[metric][idx, :] = \
                        float(sim_obs_stats[metric].values[0]), float(bcs_obs_stats[metric].values[0])

        except Exception as e:
            logger.info('failed during collecting stats')
            logger.info(e)
            errors['g4'] += 1
            continue

    bcs_nc['flow_bc'][:] = c_array
    bcs_nc['percentiles'][:] = p_array
    bcs_nc['scalars'][:] = s_array
    for idx, metric in enumerate(metric_list):
        bcs_nc[metric_nc_name_list[idx]][:] = computed_metrics[metric]

    bcs_nc.sync()
    bcs_nc.close()

    logger.info(errors)
    with open(os.path.join(workdir, 'calibration_errors.json'), 'w') as f:
        f.write(str(errors))

    return
