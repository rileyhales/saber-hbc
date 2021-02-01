import pandas as pd
import geopandas as gpd

# create a geojson showing the basins matched by data_4_assign_propagation
a = pd.read_csv('data_4_assignments/AssignmentsTable_modify_2.csv')
reasons = set(a['AssignmentReason'].dropna().tolist())
dl = gpd.read_file('data_0_inputs/magdalena_drainagelines/magdalena_drainagelines.shp')

print(len(a['AssignmentReason'].dropna().tolist()))
print(len(a['GeoglowsID'].dropna().tolist()))
exit()
for reason in reasons:
    ids = a[a['AssignmentReason'] == reason]
    dl[dl['COMID'].isin(a[a['AssignmentReason'] == reason]['GeoglowsID'].tolist())].to_file(f'ViewAssignments_{reason}.json', driver='GeoJSON')
    # dl.to_file(f'ViewAssignments_{reason}.json', driver='GeoJSON')
