import logging
import os
import statistics
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray
from natsort import natsorted
from scipy import interpolate, stats

from .fdc import fdc
from .fdc import sfdc
from .fdc import z_scale
from .io import COL_ASN_MID
from .io import COL_GID
from .io import COL_MID
from .io import COL_QMOD
from .io import COL_QOBS
from .io import COL_QSIM

logger = logging.getLogger(__name__)

__all__ = ['mp_saber', 'fdc_mapping', 'sfdc_mapping', 'map_saber']


def mp_saber(assign_df: pd.DataFrame, hindcast_zarr: str, gauge_data: str, save_dir: str = None,
             n_processes: int or None = None) -> None:
    """
    Corrects all streams in the assignment table using the SABER method with a multiprocessing Pool

    Args:
        assign_df: the assignment table
        hindcast_zarr: string path to the hindcast streamflow dataset in zarr format
        gauge_data: path to the directory of observed data
        save_dir: path to the directory to save the corrected data
        n_processes: number of processes to use for multiprocessing, passed to Pool

    Returns:
        None
    """
    logger.info('Starting SABER Bias Correction')

    if save_dir is None:
        save_dir = os.path.join(gauge_data, 'corrected')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    with Pool(n_processes) as p:
        p.starmap(
            map_saber,
            [[mid, asgn_mid, asgn_gid, hindcast_zarr, gauge_data, save_dir] for mid, asgn_mid, asgn_gid in
             np.moveaxis(assign_df[[COL_MID, COL_ASN_MID, COL_GID]].values, 0, 0)]
        )

    logger.info('Finished SABER Bias Correction')
    return


def map_saber(mid: str, asgn_mid: str, asgn_gid: str, hz: str, gauge_data: str) -> pd.DataFrame | tuple | None:
    """
    Corrects all streams in the assignment table using the SABER method

    Args:
        mid: the model id of the stream to be corrected
        asgn_mid: the model id of the stream assigned to mid for bias correction
        asgn_gid: the gauge id of the stream assigned to mid for bias correction
        hz: xarray dataset of hindcast streamflow data
        gauge_data: path to the directory of observed data

    Returns:
        None
    """
    try:
        if asgn_gid is None or pd.isna(asgn_gid):
            logger.debug(f'No gauge assigned to {mid}')
            return

        # find the observed data to be used for correction (hamro folder bata observed gauged data read garera obs_df ma rakhcha)
        if not os.path.exists(os.path.join(gauge_data, f'{asgn_gid}.csv')):
            logger.debug(f'Observed data "{asgn_gid}" not found. Cannot correct "{mid}".')
        obs_df = pd.read_csv(os.path.join(gauge_data, f'{asgn_gid}.csv'), index_col=0)  ##use this for other general file
        #obs_df = pd.read_csv(os.path.join(gauge_data, f'{asgn_gid}.csv'), index_col=0, usecols=[0, 2]).dropna(how='all')  ##only use this when running anamoly csv file
        obs_df.index = pd.to_datetime(obs_df.index)
        obs_df.iloc[:, 0] = obs_df.iloc[:, 0].clip(lower=0)
        obs_df.columns = [COL_QOBS]
        # perform corrections
        hz = xarray.open_mfdataset(hz, concat_dim='rivid', combine='nested', parallel=True, engine='zarr')
        rivids = hz.rivid.values
        sim_a = hz['Qout'][:, rivids == int(mid)].values # will extract values from xarray for all time period when rivid matches mid
        sim_a = pd.DataFrame(sim_a, index=hz['time'].values, columns=[COL_QSIM])
        #print(asgn_mid)
        #print(mid)
        if asgn_mid != mid:
            sim_b = hz['Qout'][:, rivids == int(asgn_mid)].values
            sim_b = pd.DataFrame(sim_b, index=sim_a.index, columns=[COL_QSIM])
            sim_b = sim_b[(sim_b.index.year >= 1941) & (sim_b.index.year <= 2025)]
        sim_a = sim_a[(sim_a.index.year >= 1941) & (sim_a.index.year <= 2025)]
        hz.close()

        if asgn_mid == mid:
            corrected_df = fdc_mapping(sim_a, obs_df)
        else:
            corrected_df = sfdc_mapping(
                sim_b, obs_df, sim_a,         ####is this order for the input in the function correct?
                use_log=True,
                drop_outliers = False, outlier_threshold=3,
                fit_gumbel=True, fit_range=(5, 95),
                asgn_gid = asgn_gid
            )

        return corrected_df

    except Exception as e:
        logger.error(e)
        logger.debug(f'Failed to correct {mid}')
        return


def fdc_mapping(sim_df: pd.DataFrame, obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bias corrects a dataframe of simulated values using a dataframe of observed values

    Args:
        sim_df: A dataframe with a datetime index and a single column of streamflow values
        obs_df: A dataframe with a datetime index and a single column of streamflow values

    Returns:
        pandas DataFrame with a datetime index and a single column of streamflow values
    """
    dates = []
    values = []

    for month in natsorted(sim_df.index.month.unique()):
        # filter historical data to only be current month
        month_sim = sim_df[sim_df.index.month == int(month)].dropna()
        month_obs = obs_df[obs_df.index.month == int(month)].dropna()

        # calculate the flow duration curves
        month_sim_fdc = fdc(month_sim.values)
        month_obs_fdc = fdc(month_obs.values)


        # make interpolator for 1) sim flow to sim prob, and 2) obs prob to obs flow
        to_prob = _make_interpolator(month_sim_fdc.values.flatten(), month_sim_fdc.index)
        to_flow = _make_interpolator(month_obs_fdc.index, month_obs_fdc.values.flatten())

        dates += month_sim.index.to_list()
        values += to_flow(to_prob(month_sim.values)).tolist()

    return pd.DataFrame({
        COL_QMOD: values,
        COL_QSIM: sim_df.values.flatten(),
    }, index=dates).sort_index()


def sfdc_mapping(sim_flow_a: pd.DataFrame, obs_flow_a: pd.DataFrame, sim_flow_b: pd.DataFrame = None,
                 use_log: bool = False,
                 fix_seasonally: bool = True, empty_months: str = 'skip',
                 drop_outliers: bool = False, outlier_threshold: int or float = 2.5,
                 filter_scalar_fdc: bool = False, filter_range: tuple = (0, 80),
                 extrapolate: str = 'nearest', fill_value: int or float = None,
                 fit_gumbel: bool = False, fit_range: tuple = (10, 90),
                 metadata: bool = False, asgn_gid: str = None ) -> pd.DataFrame:
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

        use_log (bool): (optional) if True, log10 transform the discharge values before correcting. default is False.

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

        metadata (bool): flag to return the scalars and metadata about the correction process

    Returns:
        pd.DataFrame with a DateTime index and columns with corrected flow, uncorrected flow, the scalar adjustment
        factor applied to correct the discharge, and the percentile of the uncorrected flow (in the seasonal grouping,
        if applicable).
    """

    def save_scalar_fdc(scalar_fdc, month_val=None):
        """Helper function to save scalar FDC for each month"""
        if asgn_gid:
            scalar_path = '/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/Bias corrected Time series/SFDC_FDC'
            os.makedirs(scalar_path, exist_ok=True)

            scalar_fdc.to_csv(os.path.join(scalar_path, f'{asgn_gid}_{month_val}.csv'))

    def adjust_scalar_fdc(scalar_fdc):
        # Create a copy of the original DataFrame to avoid altering it
        adjusted_fdc = scalar_fdc.copy()

        # Step 1: Set `scalars` to 1 if they are less than 0.001
        adjusted_fdc['scalars'] = adjusted_fdc['scalars'].apply(lambda x: 1 if x < 0.001 else x)

        # Step 2: Fill missing `p_exceed` values down to 0 with `scalars` set to 1
        full_range = pd.Index(range(0, 101))
        missing_p_exceed = full_range.difference(adjusted_fdc.index)

        if len(missing_p_exceed) > 0:  # If there are any missing values
            missing_rows = pd.DataFrame({'scalars': 1}, index=missing_p_exceed)
            adjusted_fdc = pd.concat([adjusted_fdc, missing_rows]).sort_index(ascending=False)

        return adjusted_fdc


    if fix_seasonally:
        # list of the unique months in the historical simulation. should always be 1->12 but just in case...
        monthly_results = []
        for month in sorted(set(sim_flow_a.index.strftime('%m'))):
            #print(month)
            # filter data to current iteration's month
            mon_obs_a = obs_flow_a[obs_flow_a.index.month == int(month)].dropna()

            if mon_obs_a.empty:
                if empty_months == 'skip':
                    continue
                else:
                    raise ValueError(f'Invalid value for argument "empty_months". Given: {empty_months}.')

            mon_sim_a = sim_flow_a[sim_flow_a.index.month == int(month)].dropna().clip(lower=0)
            mon_sim_b = sim_flow_b[sim_flow_b.index.month == int(month)].dropna().clip(lower=0)
            monthly_results.append(sfdc_mapping(
                mon_sim_a, mon_obs_a, mon_sim_b,
                fix_seasonally=False, empty_months=empty_months,
                drop_outliers=False, outlier_threshold=outlier_threshold,
                filter_scalar_fdc=filter_scalar_fdc, filter_range=filter_range,
                extrapolate=extrapolate, fill_value=fill_value,
                fit_gumbel=fit_gumbel, fit_range=fit_range,
                asgn_gid=asgn_gid,)
            )
        # combine the results from each monthly into a single dataframe (sorted chronologically) and return it
        return pd.concat(monthly_results).sort_index()

    if use_log:
        sim_flow_a = np.log10(sim_flow_a)
        obs_flow_a = np.log10(obs_flow_a)
        if sim_flow_b is not None:
            sim_flow_b = np.log10(sim_flow_b)

    # compute the flow duration curves
    if drop_outliers:
        sim_fdc_a = fdc(_drop_outliers_by_zscore(sim_flow_a, threshold=outlier_threshold), col_name=COL_QSIM)
        sim_fdc_b = fdc(_drop_outliers_by_zscore(sim_flow_b, threshold=outlier_threshold), col_name=COL_QSIM)
        obs_fdc = fdc(_drop_outliers_by_zscore(obs_flow_a, threshold=outlier_threshold), col_name=COL_QOBS)
    else:
        sim_fdc_a = fdc(sim_flow_a, col_name=COL_QSIM)
        sim_fdc_b = fdc(sim_flow_b, col_name=COL_QSIM)
        obs_fdc = fdc(obs_flow_a, col_name=COL_QOBS)

    # calculate the scalar flow duration curve (at point A with simulated and observed data)
    scalar_fdc = sfdc(sim_fdc_a[COL_QSIM], obs_fdc[COL_QOBS])
    scalar_fdc = adjust_scalar_fdc(scalar_fdc)
    if filter_scalar_fdc:
        scalar_fdc = scalar_fdc[scalar_fdc['p_exceed'].between(filter_range[0], filter_range[1])]

    logger.debug(f'Min/Max Scalar {scalar_fdc.min()} {scalar_fdc.max()}')

    # make interpolators: Q_b -> p_exceed_b, p_exceed_a -> scalars_a
    # flow at B converted to exceedance probabilities, then matched with the scalar computed at point A
    flow_to_percent = _make_interpolator(sim_fdc_b.values.flatten(),
                                         sim_fdc_b.index,
                                         extrap=extrapolate,
                                         fill_value=fill_value)

    percent_to_scalar = _make_interpolator(scalar_fdc.index,
                                           scalar_fdc.values.flatten(),
                                           extrap=extrapolate,
                                           fill_value=fill_value)

    # #for z scale
    # qb_original = sim_flow_b.values.flatten()
    # qb_originals, mean_flow, std_flow = z_scale(qb_original)
    # p_exceed = flow_to_percent(qb_originals)
    # scalars = percent_to_scalar(p_exceed).astype(np.float32)
    # qb_adjust = np.abs(np.divide(qb_originals, scalars))
    # qb_adjusted = (qb_adjust * std_flow) + mean_flow

    #For FDC
    qb_original = sim_flow_b.values.flatten()
    #qb_originals, mean_flow, std_flow = z_scale(qb_original)
    p_exceed = flow_to_percent(qb_original)
    scalars = percent_to_scalar(p_exceed)
    qb_adjusted = np.abs(np.divide(qb_original, scalars))

    if fit_gumbel:
        qb_adjusted = _fit_extreme_values_to_gumbel(qb_adjusted, p_exceed, fit_range)

    if use_log:
        qb_adjusted = np.power(10, qb_adjusted)
        qb_original = np.power(10, qb_original)

    response = pd.DataFrame(data=np.transpose([qb_adjusted, qb_original]),
                            index=sim_flow_b.index.to_list(),
                            columns=(COL_QMOD, COL_QSIM))
    if metadata:
        response['scalars'] = scalars
        response['p_exceed'] = p_exceed

    return response


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
    # todo check that flows are not negative and have sufficient variance - even for small variance in SAF
    # if np.max(y) - np.min(y) < 5:
    #     logger.warning('The y data has similar max/min values. You may get unanticipated results.')

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
        array of the flows with the extreme values replaced
    """
    all_values = pd.DataFrame(np.transpose([q_adjust, p_exceed]), columns=('q', 'p'))

    # compute the average and standard deviation for the values within the user specified fit_range
    mid_vals = all_values[np.logical_and(all_values['p'] >= fit_range[0], all_values['p'] <= fit_range[1])]
    xbar = statistics.mean(mid_vals['q'].values)
    std = statistics.stdev(mid_vals['q'].values, xbar)

    outlier_vals = all_values.drop(mid_vals.index)
    outlier_vals['q'] = outlier_vals['q'] = -np.log(
        -np.log(1 - (1 / (1 / (1 - (outlier_vals['p'] / 100)))))) * std * .7797 + xbar - (.45 * std)
    outlier_vals[outlier_vals < 0] = 0
    all_values.update(outlier_vals)

    return all_values['q'].values.flatten()
