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


def cluster_summary_files(workdir: str, monavg_pickle: str, fdc_pickle: str, prefix: str):
    # read the label results of the kmeans model previously stored as pickle
    ma_labels = TimeSeriesKMeans.from_pickle(monavg_pickle).labels_.tolist()
    fdc_labels = TimeSeriesKMeans.from_pickle(fdc_pickle).labels_.tolist()
    # create a dataframe showing the comid and assigned cluster number
    ids = pd.read_csv(os.path.join(workdir, 'data_inputs', f'{prefix}_fdc_normalized.csv'), index_col=0).dropna(
        axis=1).columns.tolist()
    df = pd.DataFrame(np.transpose([ids, fdc_labels, ma_labels]), columns=('ID', 'fdc_cluster', 'ma_cluster'))
    df.to_csv(os.path.join(data3, f'{prefix}_clusters.csv'), index=False)
    # create a json of the paired simulation comids
    df['ma_cluster'] = df['ma_cluster'].astype(int)
    clusters = set(sorted(df['ma_cluster'].values.tolist()))
    pairs = {}
    for i in clusters:
        pairs[i] = df[df['ma_cluster'] == i]['ID'].values.tolist()
    with open(os.path.join(data3, f'{prefix}_pairs.json'), 'w') as j:
        j.write(json.dumps(pairs))

    print('Deleting Old GeoJSONs')
    for old in glob.glob(os.path.join(data3, f'{prefix}*.geojson')):
        os.remove(old)

    print('Creating GeoJSONs')
    if prefix == 'simulated':
        gdf = gpd.read_file(
            os.path.join(data0, 'south_america-geoglows-catchment', 'south_america-geoglows-catchment.shp'))
        # gdf = gpd.read_file(os.path.join(data0, 'south_america-geoglows-drainagline', 'south_america-geoglows-drainagline.shp'))
        for cluster_number in pairs:
            savepath = os.path.join(data3, f'{prefix}_cluster_{cluster_number}.geojson')
            gdf[gdf['COMID'].isin(pairs[cluster_number])].to_file(savepath, driver='GeoJSON')
    else:
        gdf = gpd.read_file(os.path.join(data0, 'ideam_stations.json'))
        for cluster_number in pairs:
            savepath = os.path.join(data3, f'{prefix}_cluster_{cluster_number}.geojson')
            gdf[gdf['ID'].isin(pairs[cluster_number])].to_file(savepath, driver='GeoJSON')
    return
