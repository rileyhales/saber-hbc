import os

import numpy as np
import pandas as pd
import xarray as xr

from .io import COL_GID
from .io import COL_MID
from .io import COL_QSIM

__all__ = ['fdc', 'sfdc', 'precalc_sfdcs']


def fdc(flows: np.array, steps: int = 101, col_name: str = 'Q') -> pd.DataFrame:
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

# def fdc(df: pd.DataFrame, steps: int = 101, col_name: str = 'Q') -> pd.DataFrame:
#     """
#     Compute a flow duration curve (exceedance probabilities) from a time series of daily flows for a *single month*.
#
#     The function implements a year-based approach:
#         1. Group data by year, but only years with >= 20 daily records are considered valid.
#         2. If the number of valid years is > 5, compute the FDC for each valid year (0..100%),
#            then take the median across years for each percentile.
#         3. If the number of valid years is <= 5, compute a single FDC from all daily data (the original approach).
#
#     Args:
#         df (pd.DataFrame):
#             A DataFrame containing daily flow data for a single month,
#             with a DateTimeIndex and one column named `col_name`.
#             E.g., data for January from multiple years.
#         steps (int): number of steps (exceedance probabilities) to use in the FDC (default=101).
#         col_name (str): name of the column in `df` that contains the flow values (default='Q').
#
#     Returns:
#         pd.DataFrame with:
#             - index: 'p_exceed' (percent exceedance from 100 down to 0).
#             - one column: col_name (the flow values corresponding to each percentile).
#     """
#     # Percent exceedance array (e.g. 100, 99, 98, ..., 0)
#     exceed_prob = np.linspace(100, 0, steps)
#
#     # Group the data by year (assumes df has a DateTimeIndex)
#     grouped = df.groupby(df.index.year)[col_name]
#
#     # Lists to hold the FDC arrays for each valid year
#     yearly_fdcs = []
#     valid_years = []
#
#     for year, flows in grouped:
#         # Count how many valid (non-NaN) flow records exist for that year-month
#         n_valid = flows.notnull().sum()
#         # If at least 20 daily records, compute FDC
#         if n_valid >= 20:
#             fdc_flows = np.nanpercentile(flows, exceed_prob)  # shape: (steps,)
#             yearly_fdcs.append(fdc_flows)
#             valid_years.append(year)
#
#     if len(valid_years) > 5:
#         # Enough valid years => compute median across all yearly FDCs
#         # yearly_fdcs is a list of ndarrays, each shape (steps,)
#         median_fdc_flows = np.median(yearly_fdcs, axis=0)  # shape (steps,)
#         fdc_values = median_fdc_flows
#     else:
#         # Not enough valid years => fallback to the single-lumped approach
#         all_flows = df[col_name].values
#         fdc_values = np.nanpercentile(all_flows, exceed_prob)
#
#     # Build the output DataFrame
#     out_df = pd.DataFrame(fdc_values, columns=[col_name], index=exceed_prob)
#     out_df.index.name = 'p_exceed'
#     return out_df


def z_scale(flows: np.array, steps: int = 101, col_name: str = 'Q') -> pd.DataFrame:
    """
    Standardize flows (Z-scores) and return as a DataFrame with the mean and std.

    Args:
        flows: array of flows
        col_name: name of the column in the returned DataFrame

    Returns:
        pd.DataFrame with columns [col_name, 'mean_flow', 'std_flow']
    """
    mean_flow = np.mean(flows)
    std_flow = np.std(flows)

    # Standardize the flows (Z-scores)
    z_flows = (flows - mean_flow) / std_flow

    # Create a DataFrame
    exceed_prob = np.linspace(100, 0, steps)
    z_fdc_flows = np.abs(np.nanpercentile(z_flows, exceed_prob))

        # Create the DataFrame
    df = pd.DataFrame(z_fdc_flows, columns=[col_name], index=exceed_prob)
    df.index.name = 'p_exceed'

    return df

def sfdc(sim_fdc: pd.DataFrame, obs_fdc: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the scalar flow duration curve (exceedance probabilities) from two flow duration curves

    Args:
        sim_fdc: simulated flow duration curve
        obs_fdc: observed flow duration curve

    Returns:
        pd.DataFrame with index (exceedance probabilities) and a column of scalars
    """
    scalars_df = pd.DataFrame(np.divide(sim_fdc.values.flatten(),obs_fdc.values.flatten()),
        columns=['scalars', ],
        index=sim_fdc.index
    )
    scalars_df.replace(np.inf, np.nan, inplace=True)
    scalars_df.dropna(inplace=True)
    return scalars_df


def precalc_sfdcs(assign_row: pd.DataFrame, gauge_data: str, hindcast_zarr: str) -> pd.DataFrame:
    """
    Compute the scalar flow duration curve (exceedance probabilities) from two flow duration curves

    Args:
        assign_row: a single row from the assignment table
        gauge_data: string path to the directory of observed data
        hindcast_zarr: string path to the hindcast streamflow dataset

    Returns:
        pd.DataFrame with index (exceedance probabilities) and a column of scalars
    """
    # todo
    # read the simulated data
    hz = xr.open_mfdataset(hindcast_zarr, concat_dim='rivid', combine='nested', parallel=True, engine='zarr')
    sim_df = hz['Qout'][:, hz.rivid.values == int(assign_row[COL_MID])].values
    sim_df = pd.DataFrame(sim_df, index=pd.to_datetime(hz['time'].values), columns=[COL_QSIM])
    sim_df = sim_df[sim_df.index.year >= 1941]

    # read the observed data
    obs_df = pd.read_csv(os.path.join(gauge_data, f'{assign_row[COL_GID]}.csv'), index_col=0)
    obs_df.index = pd.to_datetime(obs_df.index)

    sim_fdcs = []
    obs_fdcs = []
    for month in range(1, 13):
        sim_fdcs.append(fdc(sim_df[sim_df.index.month == month].values.flatten()).values.flatten())
        obs_fdcs.append(fdc(obs_df[obs_df.index.month == month].values.flatten()).values.flatten())

    sim_fdcs.append(fdc(sim_df.values.flatten()))
    obs_fdcs.append(fdc(obs_df.values.flatten()))

    sim_fdcs = np.array(sim_fdcs)
    obs_fdcs = np.array(obs_fdcs)
    sfdcs = np.divide(sim_fdcs, obs_fdcs)
    return sfdcs
