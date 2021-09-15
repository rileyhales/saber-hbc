import os

import geopandas as gpd
import pandas as pd

from ._vocab import model_id_col
from ._vocab import reason_col


def clip_by_assignment(workdir: str, assign_table: pd.DataFrame, drain_shape: str, prefix: str = '') -> None:
    """
    Creates geojsons (in workdir/gis_outputs) for each unique value in the assignment column

    Args:
        workdir: the path to the working directory for the project
        assign_table: the assign_table dataframe
        drain_shape: path to a drainage line shapefile which can be clipped
        prefix: a prefix for names of the outputs to distinguish between data generated at separate instances

    Returns:
        None
    """
    # read the drainage line shapefile
    dl = gpd.read_file(drain_shape)
    save_dir = os.path.join(workdir, 'gis_outputs')

    # get the unique list of assignment reasons
    for reason in set(assign_table[reason_col].dropna().values):
        ids = assign_table[assign_table[reason_col] == reason][model_id_col].values
        subset = dl[dl[model_id_col].isin(ids)]
        name = f'{prefix}{"_" if prefix else ""}assignments_{reason}.json'
        subset.to_file(os.path.join(save_dir, name), driver='GeoJSON')
    return


def clip_by_ids(workdir: str, ids: list, drain_shape: str, prefix: str = '') -> None:
    """
    Creates geojsons (in workdir/gis_outputs) of the subset of 'drain_shape' with an ID in the specified list

    Args:
        workdir: path to the project directory
        ids: any iterable containing a series of model_ids
        drain_shape: path to the drainage shapefile to be clipped
        prefix: optional, a prefix to prepend to each created file's name

    Returns:
        None
    """
    dl = gpd.read_file(drain_shape)
    save_dir = os.path.join(workdir, 'gis_outputs')
    name = f'{prefix}{"_" if prefix else ""}id_subset.json'
    dl[dl[model_id_col].isin(ids)].to_file(os.path.join(save_dir, name), driver='GeoJSON')
    return


def clip_by_cluster(workdir: str, assign_table: pd.DataFrame, drain_shape: str, prefix: str = '') -> None:
    """
    Creates geojsons (in workdir/gis_outputs) of the drainage lines based on which fdc cluster they were assigned to

    Args:
        workdir: the path to the working directory for the project
        assign_table: the assign_table dataframe
        drain_shape: path to a drainage line shapefile which can be clipped
        prefix: optional, a prefix to prepend to each created file's name

    Returns:
        None
    """
    dl_gdf = gpd.read_file(drain_shape)
    ctypes = ('sim-fdc-cluster', 'obs-fdc-cluster')
    for ctype in ctypes:
        for gnum in sorted(set(assign_table[ctype].dropna().values)):
            savepath = os.path.join(workdir, 'gis_outputs', f'{prefix}{"_" if prefix else ""}{ctype}-{int(gnum)}.json')
            ids = assign_table[assign_table[ctype] == gnum][model_id_col].values
            dl_gdf[dl_gdf[model_id_col].isin(ids)].to_file(savepath, driver='GeoJSON')
    return
