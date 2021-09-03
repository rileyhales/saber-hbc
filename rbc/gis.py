import os

import numpy as np
import geopandas as gpd
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans

from ._vocab import model_id_col
from ._vocab import reason_col


def clip_by_assignment(atable: pd.DataFrame, drain_shape: str, workdir: str, prefix: str = '') -> None:
    """
    Creates geojsons in workdir/gis_outputs/assignments.

    Args:
        atable: the assign_table dataframe
        drain_shape: path to a drainageline shapefile which can be clipped
        workdir: the path to the working directory for the project
        prefix: a prefix for names of the outputs to distinguish between data generated at separate instances

    Returns:
        None
    """
    # read the drainage line shapefile
    dl = gpd.read_file(drain_shape)

    save_dir = os.path.join(workdir, 'gis_inputs', 'assignments')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # get the unique list of assignment reasons
    reasons = set(atable[reason_col].dropna().tolist())
    for reason in reasons:
        selected_segments = dl[dl[model_id_col].isin(atable[atable[reason_col] == reason][model_id_col].tolist())]
        name = f'{prefix}{"_" if prefix else ""}assignments_{reason}.json'
        selected_segments.to_file(os.path.join(save_dir, name), driver='GeoJSON')
    return


def clip_by_ids(ids: list, drain_shape: str, workdir: str, prefix: str = '') -> None:
    """
    Clips 'drain_shape' to only features whose model_id is in 'ids'

    Args:
        ids: any iterable containing a series of model_ids
        drain_shape: path to the drainage shapefile to be clipped
        workdir: path to the project directory
        prefix: optional, a prefix to prepend to each created file's name

    Returns:
        None
    """
    dl = gpd.read_file(drain_shape)
    save_dir = os.path.join(workdir, 'gis_outputs')
    name = f'{prefix}{"_" if prefix else ""}id_subset.json'
    dl[dl[model_id_col].isin(ids)].to_file(os.path.join(save_dir, name), driver='GeoJSON')
    return


