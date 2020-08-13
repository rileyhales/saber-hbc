import geopandas as gpd
import pandas as pd


def find_downstream_ids(df: pd.DataFrame, target_id: int, same_order: bool = True):
    downstream_ids = [target_id, ]
    stream_row = df[df['COMID'] == target_id]
    stream_order = stream_row['order_'].values[0]

    if same_order:
        while stream_row['NextDownID'].values[0] != -1 and stream_row['order_'].values[0] == stream_order:
            downstream_ids.append(stream_row['NextDownID'].values[0])
            stream_row = df[df['COMID'] == stream_row['NextDownID'].values[0]]
            if len(stream_row) == 0:
                break
    else:
        while stream_row['NextDownID'].values[0] != -1:
            downstream_ids.append(stream_row['NextDownID'].values[0])
            stream_row = df[df['COMID'] == stream_row['NextDownID'].values[0]]
            if len(stream_row) == 0:
                break
    return tuple(downstream_ids)


def find_upstream_ids(df: pd.DataFrame, target_id: int, same_order: bool = True):
    d = df.copy()
    if same_order:
        d = d[d['order_'] == d[d['COMID'] == target_id]['order_'].values[0]]

    upstream_ids = [target_id, ]
    upstream_rows = d[d['NextDownID'] == target_id]

    while not upstream_rows.empty or len(upstream_rows) > 0:
        if len(upstream_rows) == 1:
            upstream_ids.append(upstream_rows['COMID'].values[0])
            upstream_rows = d[d['NextDownID'] == upstream_rows['COMID'].values[0]]
        elif len(upstream_rows) > 1:
            for id in upstream_rows['COMID'].values.tolist():
                upstream_ids += list(find_upstream_ids(d, id, False))
                upstream_rows = d[d['NextDownID'] == id]
    return tuple(set(upstream_ids))


def clip_drainage_lines_by_ids(list_of_ids):
    a = gpd.read_file('/data_0_inputs/magdalena_drainagelines/south_americageoglowsdrainag.shp')
    a[a['COMID'].isin(list_of_ids)].to_file('/Users/riley/code/basin_matching/data_4_assign_propagation/clipped_lines.json', driver='GeoJSON')
    return


def propagation_assignments(df: pd.DataFrame, station: int, ids: list or tuple,
                            max_propagation: int = 5) -> pd.DataFrame:
    df_cp = df.copy()
    for distance, i in enumerate(ids):
        if distance + 1 >= max_propagation:
            print('distance is too large, no longer making assignments')
            continue
        try:
            # if the stream segment has an observation station in it, skip
            if df_cp[df_cp['GeoglowsID'] == i]['AssignmentReason'].values[0] == 'spatial':
                print('found another station, skipping to next station')
                continue
            # if the stream segment does not have an assigned value, assign it
            if pd.isna(df_cp[df_cp['GeoglowsID'] == i]['AssignedID'].values[0]):
                df_cp.loc[df_cp['GeoglowsID'] == i, 'AssignedID'] = station
                df_cp.loc[df_cp['GeoglowsID'] == i, 'AssignmentReason'] = f'Propagation-{distance + 1}'
                print('assigned')
            # if the stream segment does have an assigned value, check if this one is better before overwriting
            else:
                last_dist = int(df_cp[df_cp['GeoglowsID'] == i]['AssignmentReason'].values[0].split('-')[-1])
                print(f'last distance: {last_dist}, new distance: {distance + 1}')
                if distance + 1 < int(last_dist):
                    df_cp.loc[df_cp['GeoglowsID'] == i, 'AssignedID'] = station
                    df_cp.loc[df_cp['GeoglowsID'] == i, 'AssignmentReason'] = f'Propagation-{distance + 1}'
                    print('made better assignment')
                continue
        except Exception as e:
            print(e)
            print(f'failed to set assigned value for geoglows id {i}')
    return df_cp


def make_assignments_table():
    sim_table = pd.read_csv('/Users/riley/code/basin_matching/data_0_inputs/magdalena_table.csv')
    assignments_df = pd.DataFrame({'GeoglowsID': sim_table['COMID'].tolist(), 'Order': sim_table['order_'].tolist()})
    obs_table = pd.read_csv('/Users/riley/code/basin_matching/data_0_inputs/magdalena_stations_assignments.csv')
    assignments_df = pd.merge(assignments_df, obs_table, on='GeoglowsID', how='outer')
    assignments_df.to_csv('/Users/riley/code/basin_matching/data_4_assignments/AssignmentsTable.csv', index=False)
    return


sim_table = pd.read_csv('/Users/riley/code/basin_matching/data_0_inputs/magdalena_table.csv')
obs_table = pd.read_csv('/Users/riley/code/basin_matching/data_0_inputs/magdalena_stations_assignments.csv')
assignments_df = pd.read_csv('/Users/riley/code/basin_matching/data_4_assignments/AssignmentsTable.csv')

# for every station that we have
for station in assignments_df['StationID'].dropna():
    # determine the simulated id for the station
    geoglowsid = assignments_df[assignments_df['StationID'] == station]['GeoglowsID'].values[0]
    # identify simulated ids downstream of the station's simulated id and assign them the station id
    down_ids = find_downstream_ids(sim_table, geoglowsid, same_order=True)
    assignments_df = propagation_assignments(assignments_df, station, down_ids, max_propagation=5)
    # identify simulated ids upstream of the station's simulated id and assign them that station
    up_ids = find_upstream_ids(sim_table, geoglowsid, same_order=True)
    assignments_df = propagation_assignments(assignments_df, station, up_ids, max_propagation=5)

assignments_df.to_csv('/Users/riley/code/basin_matching/data_4_assignments/AssignmentsTable_modify.csv', index=False)
