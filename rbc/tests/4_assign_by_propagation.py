import geopandas as gpd
import pandas as pd


# 4 Assign by Propagation

make_assignments_table()

sim_table = pd.read_csv('/Users/rileyhales/code/basin_matching/data_0_inputs/magdalena_table.csv')
obs_table = pd.read_csv('/Users/rileyhales/code/basin_matching/data_0_inputs/magdalena_stations_assignments.csv')
assignments_df = pd.read_csv('/Users/rileyhales/code/basin_matching/data_4_assignments/AssignmentsTable.csv')

id_col = 'COMID'
order_col = 'order_'
next_col = 'NextDownID'

# at each station
for station in assignments_df['StationID'].dropna():
    # determine the simulated id for the basin that contains the station
    geoglowsid = assignments_df[assignments_df['StationID'] == station]['GeoglowsID'].values[0]
    # identify ids of simulated basins downstream of the station and assign them this station's id
    down_ids = walk_downstream(sim_table, geoglowsid, id_col, order_col, next_col, same_order=True)
    assignments_df = propagation_assignments(assignments_df, station, down_ids, max_propagation=5)
    # identify ids of simulated basins upstream of the station and assign them this station's id
    up_ids = walk_upstream(sim_table, geoglowsid, id_col, order_col, next_col, same_order=True)
    assignments_df = propagation_assignments(assignments_df, station, up_ids, max_propagation=5)

assignments_df.to_csv('/Users/rileyhales/code/basin_matching/data_4_assignments/AssignmentsTable_modify.csv',
                      index=False)



