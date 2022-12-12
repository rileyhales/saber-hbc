import logging
import os

import contextily as cx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import COL_ASN_REASON
from .io import COL_CID
from .io import COL_GID
from .io import COL_MID
from .io import get_dir
from .io import read_gis
from .io import read_table

__all__ = ['create_maps', 'map_by_reason', 'map_by_cluster', 'map_unassigned', 'map_ids', ]

logger = logging.getLogger(__name__)


def create_maps(assign_df: pd.DataFrame = None, drain_gis: gpd.GeoDataFrame = None, prefix: str = '') -> None:
    """
    Runs all the clip functions which create subsets of the drainage lines GIS dataset based on how they were assigned
    for bias correction.

    Args:
        assign_df: the assignment table dataframe
        drain_gis: a geodataframe of the drainage lines gis dataset
        prefix: a prefix for names of the outputs to distinguish between data generated in separate instances

    Returns:
        None
    """
    if assign_df is None:
        assign_df = read_table('assign_table')
    if drain_gis is None:
        drain_gis = read_gis('drain_gis')

    if type(drain_gis) == str:
        gdf = gpd.read_file(drain_gis)
    elif type(drain_gis) == gpd.GeoDataFrame:
        gdf = drain_gis
    else:
        raise TypeError(f'Invalid type for drain_gis: {type(drain_gis)}')

    map_by_reason(assign_df, gdf, prefix)
    map_by_cluster(assign_df, gdf, prefix)
    map_unassigned(assign_df, gdf, prefix)
    return


def map_by_reason(assign_df: pd.DataFrame, drain_gis: str or gpd.GeoDataFrame, prefix: str = '') -> None:
    """
    Creates Geopackage files in workdir/gis_outputs for each unique value in the assignment column

    Args:
        assign_df: the assignment table dataframe
        drain_gis: path to a drainage line shapefile which can be clipped
        prefix: a prefix for names of the outputs to distinguish between data generated at separate instances

    Returns:
        None
    """
    # read the drainage line shapefile
    if isinstance(drain_gis, str):
        drain_gis = gpd.read_file(drain_gis)

    # get the unique list of assignment reasons
    for reason in assign_df[COL_ASN_REASON].unique():
        logger.info(f'Creating GIS output for group: {reason}')
        selector = drain_gis[COL_MID].astype(str).isin(assign_df[assign_df[COL_ASN_REASON] == reason][COL_MID])
        subset = drain_gis[selector]
        name = f'{f"{prefix}_" if prefix else ""}assignments_{reason}.gpkg'
        if subset.empty:
            logger.debug(f'Empty filter: No streams are assigned for {reason}')
            continue
        else:
            subset.to_file(os.path.join(get_dir('gis'), name))
    return


def map_by_cluster(assign_table: pd.DataFrame, drain_gis: str, prefix: str = '') -> None:
    """
    Creates Geopackage files in workdir/gis_outputs of the drainage lines based on the fdc cluster they were assigned to

    Args:
        assign_table: the assignment table dataframe
        drain_gis: path to a drainage line shapefile which can be clipped
        prefix: optional, a prefix to prepend to each created file's name

    Returns:
        None
    """
    if isinstance(drain_gis, str):
        drain_gis = gpd.read_file(drain_gis)
    for num in assign_table[COL_CID].unique():
        logger.info(f'Creating GIS output for cluster: {num}')
        gdf = drain_gis[drain_gis[COL_MID].astype(str).isin(assign_table[assign_table[COL_CID] == num][COL_MID])]
        if gdf.empty:
            logger.debug(f'Empty filter: No streams are assigned to cluster {num}')
            continue
        gdf.to_file(os.path.join(get_dir('gis'), f'{prefix}{"_" if prefix else ""}cluster-{int(num)}.gpkg'))
    return


def map_unassigned(assign_table: pd.DataFrame, drain_gis: str, prefix: str = '') -> None:
    """
    Creates Geopackage files in workdir/gis_outputs of the drainage lines which haven't been assigned a gauge yet

    Args:
        assign_table: the assignment table dataframe
        drain_gis: path to a drainage line shapefile which can be clipped
        prefix: optional, a prefix to prepend to each created file's name

    Returns:
        None
    """
    logger.info('Creating GIS output for unassigned basins')
    if isinstance(drain_gis, str):
        drain_gis = gpd.read_file(drain_gis)
    ids = assign_table[assign_table[COL_ASN_REASON] == 'unassigned'][COL_MID].values
    subset = drain_gis[drain_gis[COL_MID].astype(str).isin(ids)]
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


def histomaps(gdf: gpd.GeoDataFrame, metric: str, prct: str) -> None:
    """
    Creates a histogram of the KGE2012 values for the validation set

    Args:
        gdf: a GeoDataFrame containing validation metrics
        metric:name of th emetric to plot
        prct: Percentile of the validation set used to generate the histogram

    Returns:
        None
    """
    core_columns = [COL_MID, COL_GID, 'geometry']
    # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # world.plot(ax=axm, color='white', edgecolor='black')

    colors = ['#dc112e', '#d6db12', '#da9707', '#13c208', '#0824c2']
    bins = [-10, 0, 0.25, 0.5, 0.75, 1]
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(boundaries=bins, ncolors=len(cmap.colors))
    title = metric.replace('KGE2012', 'Kling Gupta Efficiency 2012 - ') + f' {prct}% Gauges Excluded'

    hist_groups = []
    hist_colors = []
    categorize_by = [-np.inf, 0, 0.25, 0.5, 0.75, 1]
    for idx in range(len(categorize_by) - 1):
        gdfsub = gdf[gdf[metric] >= categorize_by[idx]]
        gdfsub = gdfsub[gdfsub[metric] < categorize_by[idx + 1]]
        if not gdfsub.empty:
            hist_groups.append(gdfsub[metric].values)
            hist_colors.append(colors[idx])

    fig, (axh, axm) = plt.subplots(
        1, 2, tight_layout=True, figsize=(9, 5), dpi=400, gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle(title, fontsize=20)

    median = round(gdf[metric].median(), 2)
    axh.set_title(f'Histogram (Median = {median})')
    axh.set_ylabel('Count')
    axh.set_xlabel('KGE 2012')
    axh.hist(hist_groups, color=hist_colors, bins=25, histtype='barstacked', edgecolor='black')
    axh.axvline(median, color='k', linestyle='dashed', linewidth=3)

    axm.set_title('Gauge Map')
    axm.set_ylabel('Latitude')
    axm.set_xlabel('Longitude')
    axm.set_xticks([])
    axm.set_yticks([])
    gdf[core_columns + [metric, ]].to_crs(epsg=3857).plot(metric)
    cx.add_basemap(ax=axm, zoom=9, source=cx.providers.Esri.WorldTopoMap, attribution='')

    fig.savefig(os.path.join(get_dir('gis'), f'{metric}_{prct}.png'))
    return
