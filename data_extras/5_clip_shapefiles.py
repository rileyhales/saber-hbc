import geopandas as gpd
import json
import os
import shutil

with open('data_4_pairbasins/pairs.json', 'r') as j:
    pairs = json.loads(j.read())

stations_gdf = gpd.read_file('/Users/riley/spatialdata/bias_correction_test_data/ideam_stations.json')
dl_gdf = gpd.read_file('/Users/riley/spatialdata/bias_correction_test_data/south_america-geoglows-catchment/south_america-geoglows-catchment.shp')

save_dir = '/Users/riley/spatialdata/bias_correction_test_data/matched_jsons/'
shutil.rmtree(save_dir)
os.mkdir(save_dir)

for i in pairs:
    a = dl_gdf[dl_gdf['COMID'].isin(pairs[i]['sim'])]
    a.to_file(os.path.join(save_dir, f"simulated_cluster{i}.geojson"), driver='GeoJSON')
    a = stations_gdf[stations_gdf['ID'].isin(pairs[i]['obs'])]
    a.to_file(os.path.join(save_dir, f"stations_cluster{i}.geojson"), driver='GeoJSON')
