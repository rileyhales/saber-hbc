import geopandas as gpd
import pandas as pd


def walk_downstream(df: pd.DataFrame, target_id: int, id_col: str, next_col: str, order_col: str = None,
                    same_order: bool = True, outlet_next_id: str or int = -1) -> tuple:
    """
    Traverse a stream network table containing a column of unique ids, a column of the id for the stream/basin
    downstream of that point, and, optionally, a column containing the stream order.

    Args:
        df (pd.DataFrame):
        target_id (int):
        id_col (str):
        next_col (str):
        order_col (str):
        same_order (bool):
        outlet_next_id (str or int):

    Returns:
        Tuple of stream ids in the order they come from the starting point.
    """
    downstream_ids = [target_id, ]

    df_ = df.copy()
    if same_order:
        df_ = df_[df_[order_col] == df_[df_[id_col] == target_id][order_col].values[0]]

    stream_row = df_[df_[id_col] == target_id]
    while stream_row[next_col].values[0] != outlet_next_id:
        downstream_ids.append(stream_row[next_col].values[0])
        stream_row = df_[df_[id_col] == stream_row[next_col].values[0]]
        if len(stream_row) == 0:
            break
    return tuple(downstream_ids)


def walk_upstream(df: pd.DataFrame, target_id: int, id_col: str, next_col: str, order_col: str = None,
                  same_order: bool = True) -> tuple:
    """
    Traverse a stream network table containing a column of unique ids, a column of the id for the stream/basin
    downstream of that point, and, optionally, a column containing the stream order.

    Args:
        df (pd.DataFrame):
        target_id (int):
        id_col (str):
        next_col (str):
        order_col (str):
        same_order (bool):

    Returns:
        Tuple of stream ids in the order they come from the starting point. If you chose same_order = False, the
        streams will appear in order on each upstream branch but the various branches will appear mixed in the tuple in
        the order they were encountered by the iterations.
    """
    df_ = df.copy()
    if same_order:
        df_ = df_[df_[order_col] == df_[df_[id_col] == target_id][order_col].values[0]]

    # start a list of the upstream ids
    upstream_ids = [target_id, ]
    upstream_rows = df_[df_[next_col] == target_id]

    while not upstream_rows.empty or len(upstream_rows) > 0:
        if len(upstream_rows) == 1:
            upstream_ids.append(upstream_rows[id_col].values[0])
            upstream_rows = df_[df_[next_col] == upstream_rows[id_col].values[0]]
        elif len(upstream_rows) > 1:
            for id in upstream_rows[id_col].values.tolist():
                upstream_ids += list(walk_upstream(df_, id, False))
                upstream_rows = df_[df_[next_col] == id]
    return tuple(set(upstream_ids))


def clip_drainage_lines_by_ids(list_of_ids):
    a = gpd.read_file('/data_0_inputs/magdalena_drainagelines/south_americageoglowsdrainag.shp')
    a[a['COMID'].isin(list_of_ids)].to_file(
        '/Users/rileyhales/code/basin_matching/data_4_assign_propagation/clipped_lines.json', driver='GeoJSON')
    return


def propagation_assignments(df: pd.DataFrame, station: int, ids: tuple, max_propagation: int = 5) -> pd.DataFrame:
    for distance, i in enumerate(ids):
        if distance + 1 >= max_propagation:
            print('distance is too large, no longer making assignments')
            continue
        try:
            # if the stream segment has an observation station in it, skip
            if df[df['GeoglowsID'] == i]['AssignmentReason'].values[0] == 'spatial':
                print('found another station, skipping to next station')
                continue
            # if the stream segment does not have an assigned value, assign it
            if pd.isna(df[df['GeoglowsID'] == i]['AssignedID'].values[0]):
                df.loc[df['GeoglowsID'] == i, 'AssignedID'] = station
                df.loc[df['GeoglowsID'] == i, 'AssignmentReason'] = f'Propagation-{distance + 1}'
                print('assigned')
            # if the stream segment does have an assigned value, check if this one is better before overwriting
            else:
                last_dist = int(df[df['GeoglowsID'] == i]['AssignmentReason'].values[0].split('-')[-1])
                print(f'last distance: {last_dist}, new distance: {distance + 1}')
                if distance + 1 < int(last_dist):
                    df.loc[df['GeoglowsID'] == i, 'AssignedID'] = station
                    df.loc[df['GeoglowsID'] == i, 'AssignmentReason'] = f'Propagation-{distance + 1}'
                    print('made better assignment')
                continue
        except Exception as e:
            print(e)
            print(f'failed to set assigned value for geoglows id {i}')
    return df


def make_assignments_table():
    sim_table = pd.read_csv('/Users/rileyhales/code/basin_matching/data_0_inputs/magdalena_table.csv')
    assignments_df = pd.DataFrame({'GeoglowsID': sim_table['COMID'].tolist(), 'Order': sim_table['order_'].tolist(),
                                   'Drainage': sim_table['Tot_Drain_'].tolist()})
    obs_table = pd.read_csv('/Users/rileyhales/code/basin_matching/data_0_inputs/magdalena_stations_assignments.csv')
    assignments_df = pd.merge(assignments_df, obs_table, on='GeoglowsID', how='outer')
    assignments_df.to_csv('/Users/rileyhales/code/basin_matching/data_4_assignments/AssignmentsTable.csv', index=False)
    return

make_assignments_table()
exit()


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
