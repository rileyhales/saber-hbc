import os
import warnings

import geopandas as gpd
import pandas as pd

from ._vocab import mid_col
from ._vocab import reason_col

__all__ = ['generate_all', 'clip_by_assignment', 'clip_by_cluster', 'clip_by_unassigned', 'clip_by_ids']


def generate_all(workdir: str, assign_table: pd.DataFrame, drain_shape: str, prefix: str = '',
                 id_column: str = mid_col) -> None:
    """
    Runs all the clip functions which create subsets of the drainage lines GIS dataset based on how they were assigned
    for bias correction.

    Args:
        workdir: the path to the working directory for the project
        assign_table: the assign_table dataframe
        drain_shape: path to a drainage line shapefile which can be clipped
        prefix: a prefix for names of the outputs to distinguish between data generated at separate instances
        id_column: name of the id column in the attributes of the shape table

    Returns:
        None
    """
    clip_by_assignment(workdir, assign_table, drain_shape, prefix, id_column)
    clip_by_cluster(workdir, assign_table, drain_shape, prefix, id_column)
    clip_by_unassigned(workdir, assign_table, drain_shape, prefix, id_column)
    return


def clip_by_assignment(workdir: str, assign_table: pd.DataFrame, drain_shape: str, prefix: str = '',
                       id_column: str = mid_col) -> None:
    """
    Creates geojsons (in workdir/gis_outputs) for each unique value in the assignment column

    Args:
        workdir: the path to the working directory for the project
        assign_table: the assign_table dataframe
        drain_shape: path to a drainage line shapefile which can be clipped
        prefix: a prefix for names of the outputs to distinguish between data generated at separate instances
        id_column: name of the id column in the attributes of the shape table

    Returns:
        None
    """
    # read the drainage line shapefile
    dl = gpd.read_file(drain_shape)
    save_dir = os.path.join(workdir, 'gis_outputs')

    # get the unique list of assignment reasons
    for reason in set(assign_table[reason_col].dropna().values):
        ids = assign_table[assign_table[reason_col] == reason][mid_col].values
        subset = dl[dl[id_column].isin(ids)]
        name = f'{prefix}{"_" if prefix else ""}assignments_{reason}.json'
        if subset.empty:
            continue
        else:
            subset.to_file(os.path.join(save_dir, name), driver='GeoJSON')
    return


def clip_by_cluster(workdir: str, assign_table: pd.DataFrame, drain_shape: str, prefix: str = '',
                    id_column: str = mid_col) -> None:
    """
    Creates GIS files (in workdir/gis_outputs) of the drainage lines based on which fdc cluster they were assigned to

    Args:
        workdir: the path to the working directory for the project
        assign_table: the assign_table dataframe
        drain_shape: path to a drainage line shapefile which can be clipped
        prefix: optional, a prefix to prepend to each created file's name
        id_column: name of the id column in the attributes of the shape table

    Returns:
        None
    """
    dl_gdf = gpd.read_file(drain_shape)
    cluster_types = [a for a in assign_table if 'cluster' in a]
    for ctype in cluster_types:
        for gnum in sorted(set(assign_table[ctype].dropna().values)):
            savepath = os.path.join(workdir, 'gis_outputs', f'{prefix}{"_" if prefix else ""}{ctype}-{int(gnum)}.json')
            ids = assign_table[assign_table[ctype] == gnum][mid_col].values
            if dl_gdf[dl_gdf[id_column].isin(ids)].empty:
                continue
            else:
                dl_gdf[dl_gdf[id_column].isin(ids)].to_file(savepath, driver='GeoJSON')
    return


def clip_by_unassigned(workdir: str, assign_table: pd.DataFrame, drain_shape: str, prefix: str = '',
                       id_column: str = mid_col) -> None:
    """
    Creates geojsons (in workdir/gis_outputs) of the drainage lines which haven't been assigned a gauge yet

    Args:
        workdir: the path to the working directory for the project
        assign_table: the assign_table dataframe
        drain_shape: path to a drainage line shapefile which can be clipped
        prefix: optional, a prefix to prepend to each created file's name
        id_column: name of the id column in the attributes of the shape table

    Returns:
        None
    """
    dl_gdf = gpd.read_file(drain_shape)
    ids = assign_table[assign_table[reason_col].isna()][mid_col].values
    subset = dl_gdf[dl_gdf[id_column].isin(ids)]
    if subset.empty:
        warnings.warn('Empty filter: No streams are unassigned')
        return
    savepath = os.path.join(workdir, 'gis_outputs', f'{prefix}{"_" if prefix else ""}assignments_unassigned.json')
    subset.to_file(savepath, driver='GeoJSON')
    return


def clip_by_ids(workdir: str, ids: list, drain_shape: str, prefix: str = '',
                id_column: str = mid_col) -> None:
    """
    Creates geojsons (in workdir/gis_outputs) of the subset of 'drain_shape' with an ID in the specified list

    Args:
        workdir: path to the project directory
        ids: any iterable containing a series of model_ids
        drain_shape: path to the drainage shapefile to be clipped
        prefix: optional, a prefix to prepend to each created file's name
        id_column: name of the id column in the attributes of the shape table

    Returns:
        None
    """
    dl = gpd.read_file(drain_shape)
    save_dir = os.path.join(workdir, 'gis_outputs')
    name = f'{prefix}{"_" if prefix else ""}id_subset.json'
    dl[dl[id_column].isin(ids)].to_file(os.path.join(save_dir, name), driver='GeoJSON')
    return


def validation_maps(workdir: str, val_table: pd.DataFrame, gauge_shape: str, prefix: str = '') -> None:
    """
    Creates geojsons (in workdir/gis_outputs) of subsets of the gauge_shape which were included in the validation tests

    Args:
        workdir: path to the project directory
        val_table: the validation table produced by rbc.validate
        gauge_shape: path to the gauge locations shapefile
        prefix: optional, a prefix to prepend to each created file's name

    Returns:
        None
    """
    save_dir = os.path.join(workdir, 'gis_outputs')

    gdf = gpd.read_file(gauge_shape)

    gdf = gdf.merge(val_table, on='gauge_id', how='inner')

    gdf.to_file(os.path.join(save_dir, 'gauges_with_validation_stats.json'), driver='GeoJSON')

    for val_set in ('50', '60', '70', '80', '90'):
        name = f'{prefix}{"_" if prefix else ""}valset_{val_set}_included.json'
        gdf[gdf['gauge_id'].isin(val_table[val_set].values.flatten())]\
            .to_file(os.path.join(save_dir, name), driver='GeoJSON')
        name = f'{prefix}{"_" if prefix else ""}valset_{val_set}_excluded.json'
        gdf[~gdf['gauge_id'].isin(val_table[val_set].values.flatten())]\
            .to_file(os.path.join(save_dir, name), driver='GeoJSON')

    return
