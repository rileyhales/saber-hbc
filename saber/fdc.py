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


def sfdc(sim_fdc: pd.DataFrame, obs_fdc: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the scalar flow duration curve (exceedance probabilities) from two flow duration curves

    Args:
        sim_fdc: simulated flow duration curve
        obs_fdc: observed flow duration curve

    Returns:
        pd.DataFrame with index (exceedance probabilities) and a column of scalars
    """
    scalars_df = pd.DataFrame(
        np.divide(sim_fdc, obs_fdc.values.flatten()),
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
    sim_df = sim_df[sim_df.index.year >= 1980]

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
