import pandas as pd
import geopandas as gpd

# create a geojson showing the basins matched by data_4_assign_propagation
a = pd.read_csv('/Users/riley/code/basin_matching/data_5_assignments/geoglowsID_stationID_assignedID.csv')
del a['StationID']
a = a.dropna()
print(a)
dl = gpd.read_file('/Users/riley/code/basin_matching/data_0_inputs/magdalena river drainagelines/south_americageoglowsdrainag.shp')
dl = dl[dl['COMID'].isin(a[a['AssignedID'] == 25027120]['GeoglowsID'].tolist())]
# dl = dl[dl['COMID'].isin(a['GeoglowsID'].tolist())]
dl.to_file('check_assignment.json', driver='GeoJSON')
