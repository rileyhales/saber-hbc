import pandas as pd
import rbc
import os
import numpy as np

def process_data (working_dir: str,
                    drainageline_csv: str,
                    catchment_csv: str,
                    drainageline_shapefile: str,
                    catchment_shapefile: str
                    ):
    """
    Takes all the necessary files outlined in the readme and creates the working directory along with the necessary
    processed csv's, shapefiles, and other files to fill it.
    Args:
        working_dir: your desired path for your working directory
        drainageline_csv: path to attribute table for geoglows-drainageline file (no need to preprocess) as csv
        catchment_csv: path to attribute table for geoglows-catchments file (no need to preprocess) as csv
        drainageline_shapefile: path to .shp file from the geoglows_drainageline folder
        catchment_shapefile: path to .shp file from the geoglows_catchments folder


    """
    #create working directory
    rbc.prep.scaffold_working_directory(working_dir)

    #retrieve, merge, and place csv's
    drain = pd.read_csv(drainageline_csv)
    catch = pd.read_csv(catchment_csv)


def get_observed_fdc(observed_data_dir: str, new_dir: str = "/Users/joshogden/Documents/regional-bias-correction-colombia_ex/examples/Australia/working_directory/data_observed"):
    """
    Takes the path to a directory containing .csvs of historical
    observed water flow over any range of time, and creates a .csv
    showing the flow duration curve for each station
    Args:
        observed_data_dir: path to directory containing observed data
            -each filename must be the station id alone
        new_dir: path to the place you want the new csv to be placed in
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


get_observed_fdc('/Users/joshogden/Documents/regional-bias-correction-colombia_ex/examples/Australia/Australia-Observed-Data')
