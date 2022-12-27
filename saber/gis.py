import logging
import os

import geopandas as gpd
import pandas as pd

from .io import COL_ASN_REASON
from .io import COL_CID
from .io import COL_MID
from .io import get_dir
from .io import read_gis
from .io import read_table

__all__ = ['create_maps', 'map_by_reason', 'map_by_cluster', 'map_unassigned', 'map_ids', 'merge_assign_table_gis']

logger = logging.getLogger(__name__)


def merge_assign_table_gis(assign_table: pd.DataFrame, gdf: str, prefix: str = '') -> None:
    """
    Creates a Geopackage file in workdir/gis of the drainage lines with the assignment table attributes added

    Args:
        assign_table: the assignment table dataframe
        gdf: path to the drainage shapefile to be clipped
        prefix: optional, a prefix to prepend to each created file's name

    Returns:
        None
    """
    if isinstance(gdf, str):
        gdf = gpd.read_file(gdf)
    gdf[COL_MID] = gdf[COL_MID].astype(str)
    name = f'{prefix}{"_" if prefix else ""}merged_assign_table_gis.gpkg'
    gdf.merge(assign_table, left_on=COL_MID, right_on=COL_MID).to_file(os.path.join(get_dir('gis'), name))
    return


def create_maps(assign_df: pd.DataFrame = None, gdf: gpd.GeoDataFrame = None, prefix: str = '') -> None:
    """
    Runs all the clip functions which create subsets of the drainage lines GIS dataset based on how they were assigned
    for bias correction.

    Args:
        assign_df: the assignment table dataframe
        gdf: a geodataframe of the drainage lines gis dataset
        prefix: a prefix for names of the outputs to distinguish between data generated in separate instances

    Returns:
        None
    """
    if assign_df is None:
        assign_df = read_table('assign_table')
    if gdf is None:
        gdf = read_gis('drain_gis')

    if type(gdf) == str:
        gdf = gpd.read_file(gdf)
    elif type(gdf) == gpd.GeoDataFrame:
        gdf = gdf
    else:
        raise TypeError(f'Invalid type for drain_gis: {type(gdf)}')

    map_by_reason(assign_df, gdf, prefix)
    map_by_cluster(assign_df, gdf, prefix)
    map_unassigned(assign_df, gdf, prefix)
    return


def map_by_reason(assign_df: pd.DataFrame, gdf: str or gpd.GeoDataFrame, prefix: str = '') -> None:
    """
    Creates Geopackage files in workdir/gis_outputs for each unique value in the assignment column

    Args:
        assign_df: the assignment table dataframe
        gdf: path to a drainage line shapefile which can be clipped
        prefix: a prefix for names of the outputs to distinguish between data generated at separate instances

    Returns:
        None
    """
    # read the drainage line shapefile
    if isinstance(gdf, str):
        gdf = gpd.read_file(gdf)

    # get the unique list of assignment reasons
    for reason in assign_df[COL_ASN_REASON].unique():
        logger.info(f'Creating GIS output for group: {reason}')
        selector = gdf[COL_MID].astype(str).isin(assign_df[assign_df[COL_ASN_REASON] == reason][COL_MID])
        subset = gdf[selector]
        name = f'{f"{prefix}_" if prefix else ""}assignments_{reason}.gpkg'
        if subset.empty:
            logger.debug(f'Empty filter: No streams are assigned for {reason}')
            continue
        else:
            subset.to_file(os.path.join(get_dir('gis'), name))
    return


def map_by_cluster(assign_table: pd.DataFrame, gdf: str, prefix: str = '') -> None:
    """
    Creates Geopackage files in workdir/gis_outputs of the drainage lines based on the fdc cluster they were assigned to

    Args:
        assign_table: the assignment table dataframe
        gdf: path to a drainage line shapefile which can be clipped
        prefix: optional, a prefix to prepend to each created file's name

    Returns:
        None
    """
    if isinstance(gdf, str):
        gdf = gpd.read_file(gdf)
    for num in assign_table[COL_CID].unique():
        logger.info(f'Creating GIS output for cluster: {num}')
        gdf = gdf[gdf[COL_MID].astype(str).isin(assign_table[assign_table[COL_CID] == num][COL_MID])]
        if gdf.empty:
            logger.debug(f'Empty filter: No streams are assigned to cluster {num}')
            continue
        gdf.to_file(os.path.join(get_dir('gis'), f'{prefix}{"_" if prefix else ""}cluster-{int(float(num))}.gpkg'))
    return


def map_unassigned(assign_table: pd.DataFrame, gdf: str, prefix: str = '') -> None:
    """
    Creates Geopackage files in workdir/gis_outputs of the drainage lines which haven't been assigned a gauge yet

    Args:
        assign_table: the assignment table dataframe
        gdf: path to a drainage line shapefile which can be clipped
        prefix: optional, a prefix to prepend to each created file's name

    Returns:
        None
    """
    logger.info('Creating GIS output for unassigned basins')
    if isinstance(gdf, str):
        gdf = gpd.read_file(gdf)
    ids = assign_table[assign_table[COL_ASN_REASON] == 'unassigned'][COL_MID].values
    subset = gdf[gdf[COL_MID].astype(str).isin(ids)]
    if subset.empty:
        logger.debug('Empty filter: No streams are unassigned')
        return
    savepath = os.path.join(get_dir('gis'), f'{prefix}{"_" if prefix else ""}assignments_unassigned.gpkg')
    subset.to_file(savepath)
    return


def map_ids(ids: list, drain_gis: str, prefix: str = '', id_column: str = COL_MID) -> None:
    """
    Creates Geopackage files in workdir/gis_outputs of the subset of 'drain_shape' with an ID in the specified list

    Args:
        ids: any iterable containing a series of model_ids
        drain_gis: path to the drainage shapefile to be clipped
        prefix: optional, a prefix to prepend to each created file's name
        id_column: name of the id column in the attributes of the shape table

    Returns:
        None
    """
    if isinstance(drain_gis, str):
        drain_gis = gpd.read_file(drain_gis)
    name = f'{prefix}{"_" if prefix else ""}id_subset.gpkg'
    drain_gis[drain_gis[id_column].isin(ids)].to_file(os.path.join(get_dir('gis'), name))
    return
