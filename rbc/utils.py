import math

import hydrostats as hs
import hydrostats.data as hd
import numpy as np
import pandas as pd


def solve_gumbel1(std, xbar, rp):
    """
    Solves the Gumbel Type I pdf = exp(-exp(-b))
    where b is the covariate
    """
    # xbar = statistics.mean(year_max_flow_list)
    # std = statistics.stdev(year_max_flow_list, xbar=xbar)
    return -math.log(-math.log(1 - (1 / rp))) * std * .7797 + xbar - (.45 * std)


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


def compute_fdc(flows: np.array, steps: int = 500, exceed: bool = True, col_name: str = 'flow'):
    percentiles = [round((1 / steps) * i * 100, 5) for i in range(steps + 1)]
    flows = np.nanpercentile(flows, percentiles)
    if exceed:
        percentiles.reverse()
    return pd.DataFrame(flows, columns=[col_name, ], index=percentiles)


def compute_scalar_fdc(first_fdc, second_fdc):
    first_fdc = compute_fdc(first_fdc)
    second_fdc = compute_fdc(second_fdc)
    ratios = np.divide(first_fdc['Flow'].values.flatten(), second_fdc['Flow'].values.flatten())
    columns = (first_fdc.columns[0], 'Scalars')
    scalars_df = pd.DataFrame(np.transpose([first_fdc.values[:, 0], ratios]), columns=columns)
    scalars_df.replace(np.inf, np.nan, inplace=True)
    scalars_df.dropna(inplace=True)

    return scalars_df
