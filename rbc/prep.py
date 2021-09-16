import os

import numpy as np
import pandas as pd
import xarray as xr

from .utils import compute_fdc


def historical_simulation(hist_nc_path: str, drain_table_path: str, save_path: str):
    # todo produce tables normalized by the maximum flows??
    """
    Process the historical simulation data netcdf into dataframes. Produces 4 tables:
    - flow duration curve for each stream
    - flow duration curve for each stream, normalized by the average flow
    - monthly averages time series for each stream
    - monthly averages time series for each stream, normalized by the average flow

    :param hist_nc_path: path to the historical simulation data netcdf
    :param drain_table_path: path to the csv of the model's drainage line attribute table
    :param save_path: a string absolute path to a directory where you want to save the csvs
    :return:
    """
    # read the drainage line table
    a = pd.read_csv(drain_table_path)
    a = a[a['order_'] > 1]
    a = sorted(a['COMID'].tolist())

    # open the historical data netcdf file
    hist_nc = xr.open_dataset(hist_nc_path)

    # start dataframes for the flow duration curve (fdc) and the monthly averages (ma) using the first comid in the list
    print('creating first dataframes')
    first_id = a.pop(0)
    first_data = hist_nc.sel(rivid=first_id).Qout.to_dataframe()['Qout']
    fdc_df = compute_fdc(
        first_data.tolist(),
        col_name=first_id
    )
    ma_df = first_data.groupby(first_data.index.strftime('%m')).mean().to_frame(name=first_id)

    # for each remaining stream ID in the list, merge/append the fdc and ma with the previously created dataframes
    print('appending more comids to initial dataframe')
    for comid in a:
        data = hist_nc.sel(rivid=comid).Qout.to_dataframe()['Qout']
        fdc_df = fdc_df.merge(compute_fdc(data.tolist(), col_name=comid),
                              how='outer', left_index=True, right_index=True)
        ma_df = ma_df.merge(data.groupby(data.index.strftime('%m')).mean().to_frame(name=comid),
                            how='outer', left_index=True, right_index=True)

    mean_annual_flow = ma_df.mean()
    fdc_df.to_pickle(os.path.join(save_path, 'simulated_fdc.csv'))
    fdc_df.div(mean_annual_flow).to_pickle(os.path.join(save_path, 'simulated_fdc_normalized.csv'))
    ma_df.to_pickle(os.path.join(save_path, 'simulated_monavg.csv'))
    ma_df.div(mean_annual_flow).to_pickle(os.path.join(save_path, 'simulated_monavg_normalized.csv'))
    # fdc_df.to_pickle(os.path.join(save_path, 'simulated_fdc.pickle'))
    # fdc_df.div(mean_annual_flow).to_pickle(os.path.join(save_path, 'simulated_fdc_normalized.pickle'))
    # ma_df.to_pickle(os.path.join(save_path, 'simulated_monavg.pickle'))
    # ma_df.div(mean_annual_flow).to_pickle(os.path.join(save_path, 'simulated_monavg_normalized.pickle'))

    return


def scaffold_working_directory(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)
    os.mkdir(os.path.join(path, 'kmeans_models'))
    os.mkdir(os.path.join(path, 'kmeans_images'))
    os.mkdir(os.path.join(path, 'data_simulated'))
    os.mkdir(os.path.join(path, 'data_observed'))


def gen_assignments_table(drain_table: str):
    sim_table = pd.read_csv(drain_table)
    assignments_df = pd.DataFrame({'GeoglowsID': sim_table['COMID'].tolist(), 'Order': sim_table['order_'].tolist(),
                                   'Drainage': sim_table['Tot_Drain_'].tolist()})
    obs_table = pd.read_csv('/Users/rileyhales/code/basin_matching/data_0_inputs/magdalena_stations_assignments.csv')
    assignments_df = pd.merge(assignments_df, obs_table, on='GeoglowsID', how='outer')
    assignments_df.to_csv('/Users/rileyhales/code/basin_matching/data_4_assignments/AssignmentsTable.csv', index=False)
    return


def observed_data(observed_data_dir: str, workdir: str):
    """
    Takes the path to a directory containing .csvs of historical observed water flow over any range of time,
    and creates a csv showing the flow duration curve for each station
    
    Args:
        observed_data_dir: path to directory containing observed data
            -each filename must be the station id alone
        workdir: path to project working directory
    
    Returns:
        None
    """
    # loop through directory and fill a dictionary with pd.DataFrames
    dict_of_df = {}

    # listofcsvs = os.listdir(observed_data_dir)  # [0,1,2,3,4]
    # firstcsv = listofcsvs.pop()  # firstcsv = 0, listofcsvs = [1,2,3,4]

    # todo: make a dataframe for the *first station
    # get a df, can be called final_df like you have below
    # final_df = pd.DataFrame(...

    # loop through the remaining stations
    for i, csv_file in enumerate(listofcsvs):
        filename = csv_file
        df_name = filename.replace('.csv', '')

        # modify to just read the data into a tmp variable instead of caching in a dictionary
        dict_of_df[df_name] = pd.read_csv(
            os.path.join(observed_data_dir, filename),
            index_col=False,
            usecols=['datetime', 'flow'],
            parse_dates=['datetime']
        )
        # todo: compute the fdc (like on line 129)

        # todo: merge the fdc dataframe into the final_df from earlier without caching in a dictionary (like line 135)

        dict_of_df[df_name] = dict_of_df[df_name].set_index('datetime')

    # loop through the dictionary and calculate the flow duration curve of each DataFrame
    fdc_dict = {}
    dict_keys = list(dict_of_df)
    dict_key_1 = dict_keys[0]
    final_df = pd.DataFrame(
        compute_fdc(
            np.array(dict_of_df[dict_key_1]['flow']), col_name=dict_key_1)
    )

    for k, df in dict_of_df:
        flows = np.array(df['flow'])
        final_df = final_df.join(compute_fdc(flows, col_name=k))

    # todo double check that the file path is right
    final_df.to_csv(os.path.join(workdir, 'data_observed', 'obs-fdc.csv'))
    return final_df
