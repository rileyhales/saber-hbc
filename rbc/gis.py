import os

import pandas as pd
import geopandas as gpd


def clip_by_assignment(assignment_table: str, drain_shape: str, save_directory: str):
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


def clip_by_ids(list_of_ids):
    a = gpd.read_file('/data_0_inputs/magdalena_drainagelines/south_americageoglowsdrainag.shp')
    a[a['COMID'].isin(list_of_ids)].to_file(
        '/Users/rileyhales/code/basin_matching/data_4_assign_propagation/clipped_lines.json', driver='GeoJSON')
    return


def cluster_summary_files(monavg_pickle: str, fdc_pickle: str, name_prefix: str):
    print('Creatings matched csv and jsons')
    # read the label results of the kmeans model previously stored as pickle
    ma_labels = TimeSeriesKMeans.from_pickle(monavg_pickle).labels_.tolist()
    fdc_labels = TimeSeriesKMeans.from_pickle(fdc_pickle).labels_.tolist()
    # create a dataframe showing the comid and assigned cluster number
    ids = pd.read_csv(os.path.join(data1, f'{name_prefix}_fdc_normalized.csv'), index_col=0).dropna(
        axis=1).columns.tolist()
    df = pd.DataFrame(np.transpose([ids, fdc_labels, ma_labels]), columns=('ID', 'fdc_cluster', 'ma_cluster'))
    df.to_csv(os.path.join(data3, f'{name_prefix}_clusters.csv'), index=False)
    # create a json of the paired simulation comids
    df['ma_cluster'] = df['ma_cluster'].astype(int)
    clusters = set(sorted(df['ma_cluster'].values.tolist()))
    pairs = {}
    for i in clusters:
        pairs[i] = df[df['ma_cluster'] == i]['ID'].values.tolist()
    with open(os.path.join(data3, f'{name_prefix}_pairs.json'), 'w') as j:
        j.write(json.dumps(pairs))

    print('Deleting Old GeoJSONs')
    for old in glob.glob(os.path.join(data3, f'{name_prefix}*.geojson')):
        os.remove(old)

    print('Creating GeoJSONs')
    if name_prefix == 'simulated':
        gdf = gpd.read_file(
            os.path.join(data0, 'south_america-geoglows-catchment', 'south_america-geoglows-catchment.shp'))
        # gdf = gpd.read_file(os.path.join(data0, 'south_america-geoglows-drainagline', 'south_america-geoglows-drainagline.shp'))
        for cluster_number in pairs:
            savepath = os.path.join(data3, f'{name_prefix}_cluster_{cluster_number}.geojson')
            gdf[gdf['COMID'].isin(pairs[cluster_number])].to_file(savepath, driver='GeoJSON')
    else:
        gdf = gpd.read_file(os.path.join(data0, 'ideam_stations.json'))
        for cluster_number in pairs:
            savepath = os.path.join(data3, f'{name_prefix}_cluster_{cluster_number}.geojson')
            gdf[gdf['ID'].isin(pairs[cluster_number])].to_file(savepath, driver='GeoJSON')
    return
