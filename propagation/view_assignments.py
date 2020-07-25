import geopandas as gpd
import pandas as pd

a = pd.read_csv('geoglows_station_assigned.csv')
dl = gpd.read_file('/Users/riley/Downloads/magdalena river drainagelines/south_americageoglowsdrainag.shp')
dl = dl[dl['COMID'].isin(a[a['AssignedID'] == 29037020.0]['GeoglowsID'].tolist())]
dl.to_file('check_assignment.json', driver='GeoJSON')


# a = pd.read_csv('geoglows_station_assigned.csv')
# for i in a.dropna()['GeoglowsID'].tolist():
#     del a[a['GeoglowsID'] == i]
# dl = gpd.read_file('/Users/riley/Downloads/magdalena river drainagelines/south_americageoglowsdrainag.shp')
# dl = dl[dl['COMID'].isin(a['GeoglowsID'].tolist())]
# dl.to_file('check_assignment.json', driver='GeoJSON')
