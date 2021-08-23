import statistics

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import interpolate, stats

from .utils import solve_gumbel1, compute_fdc, compute_scalar_fdc


def rbc_calibrate(sim_flow_a: pd.DataFrame, obs_flow_a: pd.DataFrame, sim_flow_b: pd.DataFrame,
                  fix_seasonally: bool = True, seasonality: str = 'monthly',
                  drop_outliers: bool = False, outlier_threshold: int or float = 2.5,
                  filter_scalar_fdc: bool = False, filter_range: tuple = (0, 80),
                  extrapolate_method: str = 'nearest', fill_value: int or float = None,
                  fit_gumbel: bool = False, gumbel_range: tuple = (25, 75), ) -> pd.DataFrame:
    """
    Given the simulated and observed stream flow at location a, attempts to the remove the bias from simulated
    stream flow at point b. This

    Args:
        sim_flow_a (pd.DataFrame):
        obs_flow_a (pd.DataFrame):
        sim_flow_b (pd.DataFrame):
        fix_seasonally (bool):
        seasonality (str):
        drop_outliers (bool):
        outlier_threshold (int or float):
        filter_scalar_fdc (bool):
        filter_range (tuple):
        extrapolate_method (bool):
        fill_value (int or float):
        fit_gumbel (bool):
        gumbel_range (tuple):

    Returns:

    """
    if fix_seasonally:
        if seasonality == 'monthly':
            # list of the unique months in the historical simulation. should always be 1->12 but just in case...
            monthly_results = []
            for month in sorted(set(sim_flow_a.index.strftime('%m'))):
                # filter data to only be current iteration's month
                mon_sim_data = sim_flow_a[sim_flow_a.index.month == int(month)].dropna()
                mon_obs_data = obs_flow_a[obs_flow_a.index.month == int(month)].dropna()
                mon_cor_data = sim_flow_b[sim_flow_b.index.month == int(month)].dropna()
                monthly_results.append(rbc_calibrate(
                    mon_sim_data, mon_obs_data, mon_cor_data,
                    fix_seasonally=False, seasonality=seasonality,
                    drop_outliers=drop_outliers, outlier_threshold=outlier_threshold,
                    filter_scalar_fdc=filter_scalar_fdc, filter_range=filter_range,
                    extrapolate_method=extrapolate_method, fill_value=fill_value,
                    fit_gumbel=fit_gumbel, gumbel_range=gumbel_range, )
                )
            # combine the results from each monthy into a single dataframe (sorted chronologically) and return it
            return pd.concat(monthly_results).sort_index()
        elif isinstance(seasonality, list) or isinstance(seasonality, tuple):
            # list of the unique months in the historical simulation. should always be 1->12 but just in case...
            seasonal_results = []
            for season in seasonality:
                # filter data to only be current iteration's month
                mon_sim_data = sim_flow_a[sim_flow_a.index.month.isin(season)].dropna()
                mon_obs_data = obs_flow_a[obs_flow_a.index.month.isin(season)].dropna()
                mon_cor_data = sim_flow_b[sim_flow_b.index.month.isin(season)].dropna()
                seasonal_results.append(rbc_calibrate(
                    mon_sim_data, mon_obs_data, mon_cor_data,
                    fix_seasonally=False, seasonality='monthly',
                    drop_outliers=drop_outliers, outlier_threshold=outlier_threshold,
                    filter_scalar_fdc=filter_scalar_fdc, filter_range=filter_range,
                    extrapolate_method=extrapolate_method, fill_value=fill_value,
                    fit_gumbel=fit_gumbel, gumbel_range=gumbel_range, )
                )
            return pd.concat(seasonal_results).sort_index()

    # compute the fdc for paired sim/obs data and compute scalar fdc, either with or without outliers
    if drop_outliers:
        # drop outlier data
        # https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame
        sim_fdc = compute_fdc(
            sim_flow_a[(np.abs(stats.zscore(sim_flow_a)) < outlier_threshold).all(axis=1)])
        obs_fdc = compute_fdc(
            obs_flow_a[(np.abs(stats.zscore(obs_flow_a)) < outlier_threshold).all(axis=1)])
    else:
        sim_fdc = compute_fdc(sim_flow_a)
        obs_fdc = compute_fdc(obs_flow_a)

    scalar_fdc = compute_scalar_fdc(obs_fdc['Flow'].values.flatten(), sim_fdc['Flow'].values.flatten())

    if filter_scalar_fdc:
        scalar_fdc = scalar_fdc[scalar_fdc['Exceedence Probability'] >= filter_range[0]]
        scalar_fdc = scalar_fdc[scalar_fdc['Exceedence Probability'] <= filter_range[1]]

    # Convert the percentiles
    if extrapolate_method == 'nearest':
        to_scalar = interpolate.interp1d(scalar_fdc.values[:, 0], scalar_fdc.values[:, 1],
                                         fill_value='extrapolate', kind='nearest')
    elif extrapolate_method == 'value':
        to_scalar = interpolate.interp1d(scalar_fdc.values[:, 0], scalar_fdc.values[:, 1],
                                         fill_value=fill_value, bounds_error=False)
    elif extrapolate_method == 'linear':
        to_scalar = interpolate.interp1d(scalar_fdc.values[:, 0], scalar_fdc.values[:, 1],
                                         fill_value='extrapolate')
    elif extrapolate_method == 'average':
        to_scalar = interpolate.interp1d(scalar_fdc.values[:, 0], scalar_fdc.values[:, 1],
                                         fill_value=np.mean(scalar_fdc.values[:, 1]), bounds_error=False)
    elif extrapolate_method == 'max' or extrapolate_method == 'maximum':
        to_scalar = interpolate.interp1d(scalar_fdc.values[:, 0], scalar_fdc.values[:, 1],
                                         fill_value=np.max(scalar_fdc.values[:, 1]), bounds_error=False)
    elif extrapolate_method == 'min' or extrapolate_method == 'minimum':
        to_scalar = interpolate.interp1d(scalar_fdc.values[:, 0], scalar_fdc.values[:, 1],
                                         fill_value=np.min(scalar_fdc.values[:, 1]), bounds_error=False)
    else:
        raise ValueError('Invalid extrapolation method provided')

    # determine the percentile of each uncorrected flow using the monthly fdc
    values = sim_flow_b.values.flatten()
    percentiles = [stats.percentileofscore(values, a) for a in values]
    scalars = to_scalar(percentiles)
    values = values * scalars

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

    return pd.DataFrame(data=np.transpose([values, scalars, percentiles]),
                        index=sim_flow_b.index.to_list(),
                        columns=('Propagated Corrected Streamflow', 'Scalars', 'Percentile'))


def plot_results(sim, obs, bc, bcp, title):
    sim_dates = sim.index.tolist()
    scatters = [
        go.Scatter(
            name='Propagated Corrected (Experimental)',
            x=bcp.index.tolist(),
            y=bcp['Propagated Corrected Streamflow'].values.flatten(),
        ),
        go.Scatter(
            name='Bias Corrected (Jorges Method)',
            x=sim_dates,
            y=bc.values.flatten(),
        ),
        go.Scatter(
            name='Simulated (ERA 5)',
            x=sim_dates,
            y=sim.values.flatten(),
        ),
        go.Scatter(
            name='Observed',
            x=obs.index.tolist(),
            y=obs.values.flatten(),
        ),
        go.Scatter(
            name='Percentile',
            x=sim_dates,
            y=bcp['Percentile'].values.flatten(),
        ),
        go.Scatter(
            name='Scalar',
            x=sim_dates,
            y=bcp['Scalars'].values.flatten(),
        ),
    ]
    go.Figure(scatters, layout={'title': title}).show()
    return
