import geopandas as gpd
import os

data0 = '/Users/riley/code/basin_matching/data_0_inputs'
a = gpd.read_file(os.path.join(data0, 'south_america-geoglows-catchment', 'south_america-geoglows-catchment.shp'))
b = gpd.read_file(os.path.join(data0, 'ideam_stations.json'))['GEOGLOWSID'].tolist()
ungauged = [i for i in a['COMID'].tolist() if i not in b]
a = a[a['COMID'].isin(ungauged)]
print(a)
a.to_file(os.path.join(data0, 'UngaugedBasins.geojson'), driver='GeoJSON')
