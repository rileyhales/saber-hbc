import os

import numpy as np
import pandas as pd
import xarray as xr

from .io import COL_GID
from .io import COL_MID
from .io import get_state
from .io import get_dir
from .io import read_table

__all__ = ['fdc', 'fdc_monthly', 'sfdc', 'gen_assigned_sfdcs']


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


def fdc_monthly(flows: pd.DataFrame, steps: int = 101) -> np.array:
    """
    Compute monthly flow duration curves (exceedance probabilities) from a list of flows

    Args:
        flows: array of flows
        steps: number of steps (exceedance probabilities) to use in the FDC

    Returns:
        numpy array of shape (13, steps)
        1 row per month plus an annual FDC
        1 column per exceedance probability step
    """
    fdcs = [fdc(flows[flows.index.month == m].values, steps=steps).values.flatten() for m in range(1, 13)]
    fdcs.append(fdc(flows.values, steps=steps).values.flatten())
    return np.array(fdcs)


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


def gen_assigned_sfdcs(assign_table: pd.DataFrame = None):
    """
    Compute the scalar flow duration curve (exceedance probabilities) from two flow duration curves

    Args:
        assign_table: a single row from the assignment table

    Returns:
        None
    """
    if assign_table is None:
        assign_table = read_table('assign_table')

    # select rows of the assignment table where the modeled basin contains a gauge
    at_df = assign_table[assign_table[COL_GID].notna()]

    # open the simulated data zarr
    hz = xr.open_mfdataset(get_state('hindcast_fdc_zarr'),
                           concat_dim='rivid', combine='nested', parallel=True, engine='zarr')

    # open the observed data zarr
    gz = xr.open_dataset(get_state('gauge_fdc_zarr'), engine='zarr')

    # read the simulated and oberved fdc data whose id is in the assignment table
    sim = hz.sel(rivid=at_df[COL_MID].values).to_dataframe().reset_index(drop=True)
    obs = gz.sel(gauge_id=at_df[COL_GID].values).to_dataframe().reset_index(drop=True)

    # todo check that the columns are ordered the same so the column order contains paired values

    # compute the scalar fdc
    sfdc = sim / obs
    sfdc[sfdc == np.inf] = np.nan

    # write to zarr
    xr.Dataset(
        data_vars={
            'sfdc': (('month', 'rivid', 'p'), sfdc),
        },
        coords={
            'month': np.arange(1, 14),
            'rivid': assign_table[COL_MID].values,
            'gauge_id': assign_table[COL_GID].values,
            'p': np.linspace(100, 0, 41),
        }
    ).to_zarr(os.path.join(get_dir('corrected'), 'saber_assigned_sfdc.zarr'), mode='w')

    return
