import datetime
import os
import statistics

import hydrostats as hs
import hydrostats.data as hd
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy import interpolate, stats

from ._utils import solve_gumbel1, compute_fdc, compute_scalar_fdc
from .io import asgn_gid_col
from .io import asgn_mid_col
from .io import cal_nc_name
from .io import metric_list
from .io import metric_nc_name_list
from .io import mid_col
from .io import table_hindcast


def calibrate(sim_flow_a: pd.DataFrame, obs_flow_a: pd.DataFrame, sim_flow_b: pd.DataFrame = None,
              fix_seasonally: bool = True, empty_months: str = 'skip',
              drop_outliers: bool = False, outlier_threshold: int or float = 2.5,
              filter_scalar_fdc: bool = False, filter_range: tuple = (0, 80),
              extrapolate_method: str = 'nearest', fill_value: int or float = None,
              fit_gumbel: bool = False, gumbel_range: tuple = (25, 75), ) -> pd.DataFrame:
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
        filter_scalar_fdc (bool):
        filter_range (tuple):
        extrapolate_method (str):
        fill_value (int or float):
        fit_gumbel (bool):
        gumbel_range (tuple):

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
                extrapolate_method=extrapolate_method, fill_value=fill_value,
                fit_gumbel=fit_gumbel, gumbel_range=gumbel_range, )
            )
        # combine the results from each monthly into a single dataframe (sorted chronologically) and return it
        return pd.concat(monthly_results).sort_index()

    # compute the fdc for paired sim/obs data and compute scalar fdc, either with or without outliers
    if drop_outliers:
        # drop outlier data before creating flow duration curves
        # https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
        sim_fdc = compute_fdc(
            sim_flow_a[(np.abs(stats.zscore(sim_flow_a)) < outlier_threshold).all(axis=1)])
        obs_fdc = compute_fdc(
            obs_flow_a[(np.abs(stats.zscore(obs_flow_a)) < outlier_threshold).all(axis=1)])
    else:
        sim_fdc = compute_fdc(sim_flow_a)
        obs_fdc = compute_fdc(obs_flow_a)

    scalar_fdc = compute_scalar_fdc(sim_fdc['flow'].values.flatten(), obs_fdc['flow'].values.flatten())

    if filter_scalar_fdc:
        scalar_fdc = scalar_fdc[scalar_fdc['p_exceed'].between(filter_range[0], filter_range[1])]

    # determine the percentile of each uncorrected flow using the monthly fdc
    values = sim_flow_b.values.flatten()
    percentiles = [stats.percentileofscore(values, a) for a in values]
    scalars = to_scalar(percentiles)
    values = values / scalars

    if fit_gumbel:
        tmp = pd.DataFrame(np.transpose([values, percentiles]), columns=('q', 'p'))

        # compute the average and standard deviation except for scaled data outside the percentile range specified
        mid = tmp[tmp['p'] > gumbel_range[0]]
        mid = mid[mid['p'] < gumbel_range[1]]
        xbar = statistics.mean(mid['q'].tolist())
        std = statistics.stdev(mid['q'].tolist(), xbar)

        q = []
        for p in tmp[tmp['p'] <= gumbel_range[0]]['p'].tolist():
            q.append(solve_gumbel1(std, xbar, 1 / (1 - (p / 100))))
        tmp.loc[tmp['p'] <= gumbel_range[0], 'q'] = q

        q = []
        for p in tmp[tmp['p'] >= gumbel_range[1]]['p'].tolist():
            if p >= 100:
                p = 99.999
            q.append(solve_gumbel1(std, xbar, 1 / (1 - (p / 100))))
        tmp.loc[tmp['p'] >= gumbel_range[1], 'q'] = q

        values = tmp['q'].values

    return pd.DataFrame(data=np.transpose([values, sim_flow_b.values.flatten(), scalars, percentiles]),
                        index=sim_flow_b.index.to_list(),
                        columns=('flow', 'uncorrected', 'scalars', 'percentile'))


def _make_interpolator(x_input: np.array, y_output: np.array, extrap_method: str = 'nearest',
                       const: int or float = None) -> interpolate.interp1d:
    # make interpolator which converts the percentiles to scalars
    if extrap_method == 'nearest':
        return interpolate.interp1d(x_input, y_output, fill_value='extrapolate', kind='nearest')
    elif extrap_method == 'const':
        if const is None:
            raise ValueError('Must provide the const kwarg when extrap_method="const"')
        return interpolate.interp1d(x_input, y_output, fill_value=const, bounds_error=False)
    elif extrap_method == 'linear':
        return interpolate.interp1d(x_input, y_output, fill_value='extrapolate')
    elif extrap_method == 'average':
        return interpolate.interp1d(x_input, y_output, fill_value=np.mean(y_output), bounds_error=False)
    elif extrap_method == 'max' or extrap_method == 'maximum':
        return interpolate.interp1d(x_input, y_output, fill_value=np.max(y_output), bounds_error=False)
    elif extrap_method == 'min' or extrap_method == 'minimum':
        return interpolate.interp1d(x_input, y_output, fill_value=np.min(y_output), bounds_error=False)
    else:
        raise ValueError('Invalid extrapolation method provided')


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
                print(f'\n\t\t{idx + 1}/{m_size}')

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
            print('failed to read the observed data')
            print(e)
            errors['g2'] += 1
            continue

        try:
            calibrated_df = calibrate(ts[[asgn_mid]], obs_df, ts[[model_id]], )
            c_array[:, idx] = calibrated_df['flow'].values
            p_array[:, idx] = calibrated_df['percentile'].values
            s_array[:, idx] = calibrated_df['scalars'].values
        except Exception as e:
            print('failed during the calibration step')
            print(e)
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
            print('failed during collecting stats')
            print(e)
            errors['g4'] += 1
            continue

    bcs_nc['flow_bc'][:] = c_array
    bcs_nc['percentiles'][:] = p_array
    bcs_nc['scalars'][:] = s_array
    for idx, metric in enumerate(metric_list):
        bcs_nc[metric_nc_name_list[idx]][:] = computed_metrics[metric]

    bcs_nc.sync()
    bcs_nc.close()

    print(errors)
    with open(os.path.join(workdir, 'calibration_errors.json'), 'w') as f:
        f.write(str(errors))

    return
