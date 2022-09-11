import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as Scalar

from .io import mid_col
from .io import order_col
from .io import read_table
from .io import write_table

__all__ = ['gis_tables', 'hindcast']


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
        if gauge_gis.endswith('.parquet'):
            gdf = gpd.read_parquet(gauge_gis)
        else:
            gdf = gpd.read_file(gauge_gis)
        write_table(pd.DataFrame(gdf.drop('geometry', axis=1)), workdir, 'gauge_table')

    if drain_gis is not None:
        if drain_gis.endswith('.parquet'):
            gdf = gpd.read_parquet(drain_gis)
        else:
            gdf = gpd.read_file(drain_gis)
        gdf['centroid_x'] = gdf.geometry.centroid.x
        gdf['centroid_y'] = gdf.geometry.centroid.y
        gdf = gdf.drop('geometry', axis=1)
        write_table(pd.DataFrame(gdf), workdir, 'drain_table')
    return


def hindcast(workdir: str, hind_nc_path: str, drop_order_1: bool = False) -> None:
    """
    Creates hindcast_series_table.parquet and hindcast_fdc_table.parquet in the workdir/tables directory
    for the GEOGloWS hindcast data

    Args:
        workdir: path to the working directory for the project
        hind_nc_path: path to the hindcast or historical simulation netcdf
        drop_order_1: whether to drop the order 1 streams from the hindcast

    Returns:
        None
    """
    # read the assignments table
    model_ids = read_table(workdir, 'drain_table')
    if drop_order_1:
        model_ids = model_ids[model_ids[order_col] > 1]
    model_ids = list(set(sorted(model_ids[mid_col].tolist())))

    # read the hindcast netcdf, convert to dataframe, store as parquet
    hnc = nc.Dataset(hind_nc_path)
    ids = pd.Series(hnc['rivid'][:])
    ids_selector = ids.isin(model_ids)
    ids = ids[ids_selector].astype(str).values.flatten()

    # save the model ids to table for reference
    write_table(pd.DataFrame(ids, columns=[mid_col, ]), workdir, 'model_ids')

    # save the hindcast series to parquet
    df = pd.DataFrame(
        hnc['Qout'][:, ids_selector],
        columns=ids,
        index=pd.to_datetime(hnc.variables['time'][:], unit='s')
    )
    df = df[df.index.year >= 1980]
    df.index.name = 'datetime'
    write_table(df, workdir, 'hindcast')

    # calculate the FDC and save to parquet
    exceed_prob = np.linspace(100, 0, 41)
    df = df.apply(lambda x: np.transpose(np.nanpercentile(x, exceed_prob)))
    df.index = exceed_prob
    df.index.name = 'exceed_prob'
    write_table(df, workdir, 'hindcast_fdc')

    # transform and prepare for clustering
    df = pd.DataFrame(np.transpose(Scalar().fit_transform(np.squeeze(df.values))))
    df.index = ids
    df.columns = df.columns.astype(str)
    write_table(df, workdir, 'hindcast_fdc_trans')
    return
