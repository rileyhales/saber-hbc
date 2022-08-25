import os

import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as Scalar

from ._vocab import get_table_path
from ._vocab import guess_hindcast_path
from ._vocab import mid_col
from ._vocab import read_drain_table

__all__ = ['gis_tables', 'hindcast', 'workdir']


def gis_tables(workdir: str, gauge_gis: str = None, drain_gis: str = None) -> None:
    """
    Generate copies of the drainage line attribute tables in parquet format using the Saber package vocabulary

    Args:
        workdir: path to the working directory for the project
        gauge_gis: path to the GIS dataset (e.g. geopackage) for the gauge locations (points)
        drain_gis: path to the GIS dataset (e.g. geopackage) for the drainage line locations (polylines)

    Returns:
        None
    """
    if gauge_gis is not None:
        if drain_gis.endswith('.parquet'):
            gdf = gpd.read_parquet(gauge_gis)
        else:
            gdf = gpd.read_file(gauge_gis)
        pd.DataFrame(gdf.drop('geometry', axis=1)).to_parquet(os.path.join(workdir, 'tables', 'gauge_table.parquet'))
    if drain_gis is not None:
        if drain_gis.endswith('.parquet'):
            gdf = gpd.read_parquet(drain_gis)
        else:
            gdf = gpd.read_file(drain_gis)
        gdf['centroid_x'] = gdf.geometry.centroid.x
        gdf['centroid_y'] = gdf.geometry.centroid.y
        gdf = gdf.drop('geometry', axis=1)
        pd.DataFrame(gdf).to_parquet(os.path.join(workdir, 'tables', 'drain_table.parquet'))
    return


def hindcast(workdir: str, hind_nc_path: str = None, ) -> None:
    """
    Creates hindcast_series_table.parquet.gzip and hindcast_fdc_table.parquet.gzip in the workdir/tables directory
    for the GEOGloWS hindcast data

    Args:
        workdir: path to the working directory for the project
        hind_nc_path: path to the hindcast or historical simulation netcdf if not located at workdir/data_simulated/*.nc

    Returns:
        None
    """
    if hind_nc_path is None:
        hind_nc_path = guess_hindcast_path(workdir)

    # read the assignments table
    drain_table = read_drain_table(workdir)
    model_ids = list(set(sorted(drain_table[mid_col].tolist())))

    # read the hindcast netcdf, convert to dataframe, store as parquet
    hnc = nc.Dataset(hind_nc_path)
    ids = pd.Series(hnc['rivid'][:])
    ids_selector = ids.isin(model_ids)
    ids = ids[ids_selector].astype(str).values.flatten()

    # save the model ids to table for reference
    pd.DataFrame(ids, columns=['model_id', ]).to_parquet(os.path.join(workdir, 'tables', 'model_ids.parquet'))

    # save the hindcast series to parquet
    df = pd.DataFrame(
        hnc['Qout'][:, ids_selector],
        columns=ids,
        index=pd.to_datetime(hnc.variables['time'][:], unit='s')
    )
    df = df[df.index.year >= 1980]
    df.index.name = 'datetime'
    df.to_parquet(get_table_path(workdir, 'hindcast_series'))

    # calculate the FDC and save to parquet
    exceed_prob = np.linspace(0, 100, 101)[::-1]
    df = df.apply(lambda x: np.transpose(np.nanpercentile(x, exceed_prob)))
    df.index = exceed_prob
    df.index.name = 'exceed_prob'
    df.to_parquet(get_table_path(workdir, 'hindcast_fdc'))

    # transform and prepare for clustering
    df = pd.DataFrame(np.transpose(Scalar().fit_transform(np.squeeze(df.values))))
    df.index = ids
    df.columns = df.columns.astype(str)
    df.to_parquet(os.path.join(workdir, 'tables', 'hindcast_fdc_transformed.parquet'))
    return


def workdir(path: str, include_validation: bool = True) -> None:
    """
    Creates the correct directories for a Saber project within the specified directory

    Args:
        path: the path to a directory where you want to create workdir subdirectories
        include_validation: boolean, indicates whether to create the validation folder

    Returns:
        None
    """
    dir_list = ['tables', 'data_inputs', 'gis_outputs', 'kmeans_outputs']
    if include_validation:
        dir_list.append('validation_runs')
    for d in dir_list:
        p = os.path.join(path, d)
        if not os.path.exists(p):
            os.mkdir(p)
    return
