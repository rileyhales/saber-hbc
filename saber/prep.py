import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as Scalar

from .io import write_table
from .io import mid_col

__all__ = ['gis_tables', 'calculate_fdc']


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


def calculate_fdc(workdir: str, df: pd.DataFrame, n_steps: int = 41) -> None:
    """
    Creates the hindcast_fdc.parquet and hindcast_fdc_transformed.parquet tables in the workdir/tables directory

    Args:
        workdir: path to the working directory for the project
        df: the hindcast hydrograph data DataFrame with 1 column per stream, 1 row per timestep, string column names
            containing the stream's ID, and a datetime index. E.g. the shape should be (n_timesteps, n_streams). If not
            provided, the function will attempt to load the data from workdir/tables/hindcast_series_table.parquet
        n_steps: the number of exceedance probabilities to estimate from 0 to 100%, inclusive. Default is 41, which
            produces, 0, 2.5, 5, ..., 97.5, 100.

    Returns:
        None
    """
    exceed_prob = np.linspace(100, 0, n_steps)

    # write the ID list to file
    write_table(pd.DataFrame(df.columns, columns=[mid_col, ]), workdir, 'model_ids')

    # calculate the FDC and save to parquet
    fdc_df = df.apply(lambda x: np.transpose(np.nanpercentile(x, exceed_prob)))
    fdc_df.index = exceed_prob
    fdc_df.index.name = 'exceed_prob'
    write_table(fdc_df, workdir, 'hindcast_fdc')

    # transform and prepare for clustering
    fdc_df = pd.DataFrame(np.transpose(Scalar().fit_transform(np.squeeze(fdc_df.values))))
    fdc_df.index = df.columns
    fdc_df.columns = fdc_df.columns.astype(str)
    write_table(fdc_df, workdir, 'hindcast_fdc_trans')
    return
