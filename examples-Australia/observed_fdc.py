import pandas as pd
import rbc
import os
import numpy as np

def get_observed_fdc(observed_data_dir: str, new_dir: str = "/Users/joshogden/Documents/regional-bias-correction-colombia_ex/examples/Australia/working_directory/data_observed"):
    """
    Takes the path to a directory containing .csvs of historical
    observed water flow over any range of time, and creates a .csv
    showing the flow duration curve for each station
    Args:
        observed_data_dir: path to directory containing observed data
            -each filename must be the station id alone
        new_dir: path to the directory which which you want the new file to be placed
    Returns: none
    """
    #loop through directory and fill a dictionary with pd.DataFrames
    dict_of_df = {}
    cnt = len(os.listdir(observed_data_dir))
    for i in range(0,cnt):
        filename = os.listdir(observed_data_dir)[i]
        df_name = filename.replace('.csv', '')

        dict_of_df[f'{df_name}'] = pd.read_csv(
                os.path.join(observed_data_dir, filename),
                index_col = False,
                usecols= ['datetime','flow'],
                parse_dates= ['datetime']
                )
        dict_of_df[f'{df_name}'] = dict_of_df[f'{df_name}'].set_index('datetime')

    #loop through the dictionary and calculate the flow duration curve of each DataFrame
    fdc_dict = {}
    dict_keys = list(dict_of_df)
    dict_key_1 = dict_keys[0]
    final_df = pd.DataFrame(rbc.utils.compute_fdc(np.array(dict_of_df[dict_key_1]['flow']), col_name = dict_key_1))

    for i in range(1, len(dict_keys)):
        flows = np.array(dict_of_df[dict_keys[i]]['flow'])
        final_df = final_df.join(rbc.utils.compute_fdc(flows, col_name = dict_keys[i]))
    final_df.to_csv(os.path.join(new_dir, 'obs_fdc.csv'))
    return final_df
