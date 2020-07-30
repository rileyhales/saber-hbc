import math
import statistics
from io import StringIO

import geoglows
import hydrostats as hs
import hydrostats.data as hd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from scipy import interpolate
from scipy import stats


def collect_data(start_id, start_ideam_id, downstream_id, downstream_ideam_id):
    # Upstream simulated flow
    start = geoglows.streamflow.historic_simulation(start_id)
    # Upstream observed flow
    start_ideam = get_ideam_flow(start_ideam_id)
    start_ideam.dropna(inplace=True)

    # Downstream simulated flow
    downstream = geoglows.streamflow.historic_simulation(downstream_id)
    # Downstream observed flow
    downstream_ideam = get_ideam_flow(downstream_ideam_id)
    downstream_ideam.dropna(inplace=True)
    # Downstream bias corrected flow (for comparison to the data_4_assign_propagation method
    downstream_bc = geoglows.bias.correct_historical(downstream, downstream_ideam)

    # Export all as csv
    start.to_csv('start_flow.csv')
    start_ideam.to_csv('start_ideam_flow.csv')
    downstream.to_csv('downstream_flow.csv')
    downstream_ideam.to_csv('downstream_ideam_flow.csv')
    downstream_bc.to_csv('downstream_bc_flow.csv')
    return


def get_ideam_flow(id):
    # get the gauged data
    url = f'https://www.hydroshare.org/resource/d222676fbd984a81911761ca1ba936bf/' \
          f'data/contents/Discharge_Data/{id}.csv'
    df = pd.read_csv(StringIO(requests.get(url).text), index_col=0)
    df.index = pd.to_datetime(df.index).tz_localize('UTC')
    return df


def compute_flow_duration_curve(hydro: list or np.array, prob_steps: int = 500, exceedence: bool = True):
    percentiles = [round((1 / prob_steps) * i * 100, 5) for i in range(prob_steps + 1)]
    flows = np.nanpercentile(hydro, percentiles)
    if exceedence:
        percentiles.reverse()
        columns = ['Exceedence Probability', 'Flow']
    else:
        columns = ['Non-Exceedence Probability', 'Flow']
    return pd.DataFrame(np.transpose([percentiles, flows]), columns=columns)


def get_scalar_bias_fdc(first_series, seconds_series):
    first_fdc = compute_flow_duration_curve(first_series)
    second_fdc = compute_flow_duration_curve(seconds_series)
    ratios = np.divide(first_fdc['Flow'].values.flatten(), second_fdc['Flow'].values.flatten())
    columns = (first_fdc.columns[0], 'Scalars')
    scalars_df = pd.DataFrame(np.transpose([first_fdc.values[:, 0], ratios]), columns=columns)
    scalars_df.replace(np.inf, np.nan, inplace=True)
    scalars_df.dropna(inplace=True)

    return scalars_df


def solve_gumbel_flow(std, xbar, rp):
    """
    Solves the Gumbel Type I pdf = exp(-exp(-b))
    where b is the covariate
    """
    # xbar = statistics.mean(year_max_flow_list)
    # std = statistics.stdev(year_max_flow_list, xbar=xbar)
    return -math.log(-math.log(1 - (1 / rp))) * std * .7797 + xbar - (.45 * std)


def propagate_correction(sim_flow_a: pd.DataFrame, obs_flow_a: pd.DataFrame, sim_flow_b: pd.DataFrame,
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
                monthly_results.append(propagate_correction(
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
                seasonal_results.append(propagate_correction(
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
        sim_fdc = compute_flow_duration_curve(
            sim_flow_a[(np.abs(stats.zscore(sim_flow_a)) < outlier_threshold).all(axis=1)])
        obs_fdc = compute_flow_duration_curve(
            obs_flow_a[(np.abs(stats.zscore(obs_flow_a)) < outlier_threshold).all(axis=1)])
    else:
        sim_fdc = compute_flow_duration_curve(sim_flow_a)
        obs_fdc = compute_flow_duration_curve(obs_flow_a)

    scalar_fdc = get_scalar_bias_fdc(obs_fdc['Flow'].values.flatten(), sim_fdc['Flow'].values.flatten())

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
            q.append(solve_gumbel_flow(std, xbar, 1 / (1 - (p / 100))))
        tmp.loc[tmp['p'] <= gumbel_range[0], 'q'] = q

        q = []
        for p in tmp[tmp['p'] >= gumbel_range[1]]['p'].tolist():
            if p >= 100:
                p = 99.999
            q.append(solve_gumbel_flow(std, xbar, 1 / (1 - (p / 100))))
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


def statistics_tables(corrected: pd.DataFrame, simulated: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    # merge the datasets together
    merged_sim_obs = hd.merge_data(sim_df=simulated, obs_df=observed)
    merged_cor_obs = hd.merge_data(sim_df=corrected, obs_df=observed)

    metrics = ['ME', 'RMSE', 'NRMSE (Mean)', 'MAPE', 'NSE', 'KGE (2009)', 'KGE (2012)']
    # Merge Data
    table1 = hs.make_table(merged_dataframe=merged_sim_obs, metrics=metrics)
    table2 = hs.make_table(merged_dataframe=merged_cor_obs, metrics=metrics)

    table2 = table2.rename(index={'Full Time Series': 'Corrected Full Time Series'})
    table1 = table1.rename(index={'Full Time Series': 'Original Full Time Series'})
    table1 = table1.transpose()
    table2 = table2.transpose()

    return pd.merge(table1, table2, right_index=True, left_index=True)


# collect_data(9012999, 22057070, 9012650, 22057010)
# collect_data(9017261, 32037030, 9015333, 32097010)  # really long range down stream
# collect_data(9009660, 21237020, 9007292, 23097040)  # large river
# collect_data(9007292, 23097040, 9009660, 21237020)  # large river backwards (going upstream)

# Read all as csv
start_flow = pd.read_csv('data_4_assign_propagation/start_flow.csv', index_col=0)
start_ideam_flow = pd.read_csv('data_4_assign_propagation/start_ideam_flow.csv', index_col=0)
downstream_flow = pd.read_csv('data_4_assign_propagation/downstream_flow.csv', index_col=0)
downstream_ideam_flow = pd.read_csv('data_4_assign_propagation/downstream_ideam_flow.csv', index_col=0)
downstream_bc_flow = pd.read_csv('data_4_assign_propagation/downstream_bc_flow.csv', index_col=0)
start_flow.index = pd.to_datetime(start_flow.index)
start_ideam_flow.index = pd.to_datetime(start_ideam_flow.index)
downstream_flow.index = pd.to_datetime(downstream_flow.index)
downstream_ideam_flow.index = pd.to_datetime(downstream_ideam_flow.index)
downstream_bc_flow.index = pd.to_datetime(downstream_bc_flow.index)

downstream_prop_correct = propagate_correction(start_flow, start_ideam_flow, downstream_flow,
                                               fit_gumbel=True, gumbel_range=(25, 75))
plot_results(downstream_flow, downstream_ideam_flow, downstream_bc_flow, downstream_prop_correct,
             f'Correct Monthly - Force Gumbel Distribution')
del downstream_prop_correct['Scalars'], downstream_prop_correct['Percentile']
statistics_tables(downstream_prop_correct, downstream_flow, downstream_ideam_flow).to_csv(
    'data_4_assign_propagation/stats_test.csv')
