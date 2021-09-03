import os

import numpy as np
import pandas as pd
import xarray as xr

from .utils import compute_fdc

from ._vocab import model_id_col
from ._vocab import order_col


def historical_simulation(hist_nc_path: str,
                          working_dir: str, ):
    """
    Fills the working_dir/data_simulated directory with information from the historical simulation netcdf file

    Process the historical simulation data netcdf into dataframes. Produces 4 tables:
        - flow duration curve for each stream
        - flow duration curve for each stream, normalized by the average flow
        - monthly averages time series for each stream
        - monthly averages time series for each stream, normalized by the average flow

    Args:
        hist_nc_path: path to the historical simulation data netcdf
        working_dir: path to the working directory for the project

    Returns:
        None
    """
    # read the drainage line table
    a = pd.read_csv(os.path.join(working_dir, 'assign_table.csv'))
    a = a[a[order_col] > 1]
    a = sorted(a[model_id_col].tolist())

    # open the historical data netcdf file
    hist_nc = xr.open_dataset(hist_nc_path)

    # start dataframes for the flow duration curve (fdc) and the monthly averages (ma) using the first comid in the list
    first_id = a.pop(0)
    first_data = hist_nc.sel(rivid=first_id).Qout.to_dataframe()['Qout']
    fdc_df = compute_fdc(first_data.tolist(), col_name=first_id)
    ma_df = first_data.groupby(first_data.index.strftime('%m')).mean().to_frame(name=first_id)

    # for each remaining stream ID in the list, merge/append the fdc and ma with the previously created dataframes
    for model_id in a:
        data = hist_nc.sel(rivid=model_id).Qout.to_dataframe()['Qout']
        fdc_df = fdc_df.merge(compute_fdc(data.tolist(), col_name=model_id),
                              how='outer', left_index=True, right_index=True)
        ma_df = ma_df.merge(data.groupby(data.index.strftime('%m')).mean().to_frame(name=model_id),
                            how='outer', left_index=True, right_index=True)

    mean_annual_flow = ma_df.mean()
    sim_data_path = os.path.join(working_dir, 'data_simulated')

    fdc_df.to_csv(os.path.join(sim_data_path, 'sim-fdc.csv'))
    fdc_df.to_pickle(os.path.join(sim_data_path, 'sim-fdc.pickle'))
    ma_df.to_csv(os.path.join(sim_data_path, 'sim-monavg.csv'))
    ma_df.to_pickle(os.path.join(sim_data_path, 'sim-monavg.pickle'))
    fdc_df.div(mean_annual_flow).to_csv(os.path.join(sim_data_path, 'sim-fdc-norm.csv'))
    fdc_df.div(mean_annual_flow).to_pickle(os.path.join(sim_data_path, 'sim-fdc-norm.pickle'))
    ma_df.div(mean_annual_flow).to_csv(os.path.join(sim_data_path, 'sim-monavg-norm.csv'))
    ma_df.div(mean_annual_flow).to_pickle(os.path.join(sim_data_path, 'sim-monavg-norm.pickle'))

    return


def scaffold_working_directory(path: str):
    """
    Creates the correct directories for an RBC project within the a specified directory

    Args:
        path: the path to a directory where you want to create directories

    Returns:
        None
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    os.mkdir(os.path.join(path, 'kmeans_models'))
    os.mkdir(os.path.join(path, 'kmeans_images'))
    os.mkdir(os.path.join(path, 'data_simulated'))
    os.mkdir(os.path.join(path, 'data_observed'))
    os.mkdir(os.path.join(path, 'gis_inputs'))
    os.mkdir(os.path.join(path, 'gis_outputs'))
    return


def gen_assignments_table(working_dir):
    """
    Joins the drain_table.csv and gauge_table.csv to create the assign_table.csv

    Args:
        working_dir: path to the working directory

    Returns:
        None
    """
    drain_table = os.path.join(working_dir, 'gis_inputs', 'drain_table.csv')
    gauge_table = os.path.join(working_dir, 'gis_inputs', 'gauge_table.csv')
    drain_df = pd.read_csv(drain_table, index_col=0)
    gauge_df = pd.read_csv(gauge_table, index_col=0)
    assign_table_path = os.path.join(working_dir, 'assign_table.csv')
    assign_table = pd.merge(drain_df, gauge_df, on=model_id_col, how='outer')
    assign_table['assigned_id'] = np.nan
    assign_table['reason'] = np.nan
    assign_table.to_csv(assign_table_path, index=False)
    return
