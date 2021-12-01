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
    Creates sim-fdc.csv and sim_time_series.pickle in workdir/data_processed for geoglows historical simulation data

    Args:
        workdir: path to the working directory for the project
        hist_nc_path: path to the historical simulation netcdf if not located at workdir/data_simulated/*.nc

    Returns:
        None
    """
    if hist_nc_path is None:
        hist_nc_path = guess_hist_sim_path(workdir)

    # read the assignments table
    drain_table = pd.read_csv(os.path.join(workdir, 'gis_inputs', 'drain_table.csv'))
    model_ids = list(set(sorted(drain_table[mid_col].tolist())))

    # open the historical data netcdf file
    hist_nc = xr.open_dataset(hist_nc_path)

    # start dataframes for the flow duration curve (fdc) and the monthly averages (ma) using the first comid in the list
    first_id = model_ids.pop(0)
    first_data = hist_nc.sel(rivid=first_id).Qout.to_dataframe()['Qout']
    fdc_df = compute_fdc(first_data.tolist(), col_name=first_id)
    # ma_df = first_data.groupby(first_data.index.strftime('%m')).mean().to_frame(name=first_id)

    # for each remaining stream ID in the list, merge/append the fdc and ma with the previously created dataframes
    for model_id in model_ids:
        data = hist_nc.sel(rivid=model_id).Qout.to_dataframe()['Qout']
        fdc_df = fdc_df.merge(compute_fdc(data.tolist(), col_name=model_id),
                              how='outer', left_index=True, right_index=True)

    fdc_df.to_csv(os.path.join(workdir, 'data_processed', 'sim-fdc.csv'))

    # create the time series table of simulated flow values
    coords = drain_table[[mid_col]]
    coords.loc[:, 'time'] = None
    coords = coords[['time', 'model_id']].values.tolist()
    ts = grids.TimeSeries([hist_nc_path, ], 'Qout', ('time', 'rivid'))
    ts = ts.multipoint(*coords)
    ts.set_index('datetime', inplace=True)
    ts.index = pd.to_datetime(ts.index, unit='s')
    ts.columns = drain_table[mid_col].values.flatten()
    ts.index = ts.index.tz_localize('UTC')
    ts.to_pickle(os.path.join(workdir, 'data_processed', 'sim_time_series.pickle')
)
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
