import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

from .utils import compute_fdc

from ._vocab import model_id_col
from .table import read as read_table


def historical_simulation(hist_nc_path: str, workdir: str, ) -> None:
    """
    Fills the working_dir/data_simulated directory with information from the historical simulation netcdf file

    Process the historical simulation data netcdf into dataframes. Produces 4 tables:
        - flow duration curve for each stream
        - flow duration curve for each stream, normalized by the average flow
        - monthly averages time series for each stream
        - monthly averages time series for each stream, normalized by the average flow

    Args:
        hist_nc_path: path to the historical simulation data netcdf
        workdir: path to the working directory for the project

    Returns:
        None
    """
    # read the assignments table
    a = read_table(workdir)
    a = set(sorted(a[model_id_col].tolist()))
    a = list(a)

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

    sim_data_path = os.path.join(workdir, 'data_simulated')
    fdc_df.to_csv(os.path.join(sim_data_path, 'sim-fdc.csv'))
    ma_df.to_csv(os.path.join(sim_data_path, 'sim-monavg.csv'))

    return


def observed_data(obs_dir: str, workdir: str) -> None:
    """
    Takes the path to a directory containing csv data of observed discharge over any range of time and creates
    a csv showing the flow duration curve for each station
    
    Args:
        obs_dir: path to directory containing observed data csv files, each csv named: <station_number>.csv
        workdir: path to project working directory
    
    Returns:
        None
    """
    # create a list of file names and pop out the first station csv
    csvs = glob.glob(os.path.join(obs_dir, '*.csv'))
    csvs = list(csvs)

    first_csv = csvs.pop(0)
    first_id = os.path.splitext(os.path.basename(first_csv))[0]

    # make a dataframe for the first station
    first_station = pd.read_csv(
        first_csv,
        index_col=0,
    )

    # initialize final_df
    final_df = pd.DataFrame(
        compute_fdc(
            first_station.values.flatten(),
            col_name=first_id
        )
    )

    # loop through the remaining stations
    for csv in csvs:
        station_id = os.path.splitext(os.path.basename(csv))[0]

        # read data into a temporary df
        tmp_df = pd.read_csv(
            csv,
            index_col=0,
        )
    final_df = final_df.join(compute_fdc(tmp_df.values.flatten(), col_name=station_id))

    final_df.to_csv(os.path.join(workdir, 'data_observed', 'obs-fdc.csv'))
    return


def scaffold_working_directory(path: str) -> None:
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


def gen_assignments_table(working_dir) -> pd.DataFrame:
    """
    Joins the drain_table.csv and gauge_table.csv to create the assign_table.csv

    Args:
        working_dir: path to the working directory

    Returns:
        None
    """
    drain_table = os.path.join(working_dir, 'gis_inputs', 'drain_table.csv')
    gauge_table = os.path.join(working_dir, 'gis_inputs', 'gauge_table.csv')
    drain_df = pd.read_csv(drain_table)
    gauge_df = pd.read_csv(gauge_table)
    assign_table = pd.merge(drain_df, gauge_df, on=model_id_col, how='outer')
    assign_table['assigned_id'] = np.nan
    assign_table['reason'] = np.nan
    return assign_table
