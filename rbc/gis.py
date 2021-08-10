import os

import pandas as pd
import geopandas as gpd


def assignment_reason(assignment_table: str, drain_shape: str, save_directory: str):
    """
    Creates multiple geojson, each contains a basins assigned a gauge for the same reason

    :param assignment_table: path to the assignment table
    :param drain_shape: path to the drainageline shapefile
    :return:
    """
    # read the assignments table
    a = pd.read_csv(assignment_table)
    # read the drainage line shapefile
    dl = gpd.read_file(drain_shape)
    # get the unique list of assignment reasons
    reasons = set(a['AssignmentReason'].dropna().tolist())
    for reason in reasons:
        dl[dl['COMID'].isin(a[a['AssignmentReason'] == reason]['GeoglowsID'].tolist())].to_file(
            os.path.join(save_directory, f'Assignment-{reason}.json'), driver='GeoJSON')
    return
