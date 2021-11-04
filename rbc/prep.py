import glob
import os

import grids
import pandas as pd
import xarray as xr

from .utils import compute_fdc

from ._vocab import mid_col


def guess_hist_sim_path(workdir: str) -> str:
    return glob.glob(os.path.join(workdir, 'data_inputs', '*.nc*'))[0]


def historical_simulation(workdir: str, hist_nc_path: str = None) -> None:
    """
    Fills the working_dir/data_simulated directory with information from the historical simulation netcdf file

    Process the historical simulation data netcdf into dataframes. Produces 4 tables:
        - flow duration curve for each stream
        - flow duration curve for each stream, normalized by the average flow
        - monthly averages time series for each stream
        - monthly averages time series for each stream, normalized by the average flow

    Args:
        workdir: path to the working directory for the project
        hist_nc_path: path to the historical simulation netcdf if not located at workdir/data_simulated/*.nc

    Returns:
        None
    """
    if hist_nc_path is None:
        hist_nc_path = guess_hist_sim_path(workdir)

    # read the assignments table
    a = pd.read_csv(os.path.join(workdir, 'gis_inputs', 'drain_table'))
    a = list(set(sorted(a[mid_col].tolist())))

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

    data_out_path = os.path.join(workdir, 'data_processed')
    fdc_df.to_csv(os.path.join(data_out_path, 'sim-fdc.csv'))
    ma_df.to_csv(os.path.join(data_out_path, 'sim-monavg.csv'))
    return


def hist_sim_table(workdir: str, hist_nc_path: str = None) -> None:
    if hist_nc_path is None:
        hist_nc_path = guess_hist_sim_path(workdir)
    ts_table = os.path.join(workdir, 'data_processed', 'subset_time_series.pickle')
    drain_table = pd.read_csv(os.path.join(workdir, 'gis_inputs', 'drain_table.csv'))

    # get the simulated values and coordinate variables
    coords = drain_table[[mid_col]]
    coords.loc[:, 'time'] = None
    coords = coords[['time', 'model_id']].values.tolist()
    ts = grids.TimeSeries([hist_nc_path, ], 'Qout', ('time', 'rivid'))
    ts = ts.multipoint(*coords)
    ts.set_index('datetime', inplace=True)
    ts.index = pd.to_datetime(ts.index, unit='s')
    ts.columns = drain_table['model_id'].values.flatten()
    ts.to_pickle(ts_table)
    return


def observed_data(workdir: str, obs_data_path: str = None) -> None:
    """
    Takes the path to a directory containing csv data of observed discharge over any range of time and creates
    a csv showing the flow duration curve for each station
    
    Args:
        workdir: path to project working directory
        obs_data_path: path to the observed data directory if not located at workdir/data_observed/csvs

    Returns:
        None
    """
    if obs_data_path is None:
        obs_data_path = os.path.join(workdir, 'data_inputs', 'obs_csvs')

    # create a list of file names and pop out the first station csv
    csvs = glob.glob(os.path.join(obs_data_path, '*.csv'))
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

    final_df.to_csv(os.path.join(workdir, 'data_processed', 'obs-fdc.csv'))
    return


def scaffold_workdir(path: str, include_validation: bool = True) -> None:
    """
    Creates the correct directories for an RBC project within the a specified directory

    Args:
        path: the path to a directory where you want to create directories
        include_validation: boolean, indicates whether or not to create the validation folder

    Returns:
        None
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    os.mkdir(os.path.join(path, 'data_inputs'))
    os.mkdir(os.path.join(path, 'data_processed'))
    os.mkdir(os.path.join(path, 'gis_inputs'))
    os.mkdir(os.path.join(path, 'gis_outputs'))
    os.mkdir(os.path.join(path, 'kmeans_models'))
    os.mkdir(os.path.join(path, 'kmeans_images'))
    if include_validation:
        os.mkdir(os.path.join(path, 'validation_runs'))
    return
