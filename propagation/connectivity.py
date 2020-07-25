import pandas as pd
import numpy as np
import geopandas as gpd


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
    upstream_ids = [target_id, ]
    stream_order = df[df['COMID'] == target_id]['order_'].values[0]

    current_id = target_id

    while True:
        if same_order:
            upstream_rows = df[df['NextDownID'] == current_id]
            upstream_rows = upstream_rows[upstream_rows['order_'] == stream_order]
        else:
            upstream_rows = df[df['NextDownID'] == current_id]
        if len(upstream_rows) == 0:
            return set(tuple(upstream_ids))
        if len(upstream_rows) == 1:
            current_id = upstream_rows['COMID'].values[0]
            upstream_ids.append(current_id)
        elif len(upstream_rows) > 1:
            for id in upstream_rows['COMID'].values.tolist():
                current_id = id
                asdf = find_upstream_ids(df, id, same_order=False)
                upstream_ids += list(asdf)
    return tuple(set(upstream_ids))


gsa_df = pd.read_csv('magdalena_table.csv')['COMID'].tolist()
gsa_df = pd.DataFrame(np.transpose(gsa_df), columns=['GeoglowsID'])
b = pd.read_csv('magdalena_stations_table.csv')
gsa_df = pd.merge(gsa_df, b, on='GeoglowsID', how='outer')
gsa_df.to_csv('geoglows_station_assigned.csv', index=False)
gsa_df.fillna(0, inplace=True)
print(gsa_df)

b = pd.read_csv('magdalena_table.csv')

for station in gsa_df['StationID'].dropna():
    # find list of down stream segments
    geoglowsid = gsa_df[gsa_df['StationID'] == station]['GeoglowsID'].values[0]
    ids = find_downstream_ids(b, geoglowsid)
    for i in ids:
        # the downstream segment doesn't have an assigned idea, assign it the current station
        try:
            if gsa_df[gsa_df['GeoglowsID'] == i]['AssignedID'].values[0] == 0:
                gsa_df.loc[gsa_df['GeoglowsID'] == i, 'AssignedID'] = station
                gsa_df.loc[gsa_df['GeoglowsID'] == i, 'AssignmentReason'] = 'propagation'
                print('assigned')
        except:
            print(f'failed to set assigned value for geoglows id {i}')
    # ids = find_upstream_ids(b, geoglowsid)
    # for i in ids:
    #     the downstream segment doesn't have an assigned idea, assign it the current station
        # if gsa_df[gsa_df['GeoglowsID'] == i]['AssignedID'].values[0] == np.nan:
        #     gsa_df.loc[gsa_df['GeoglowsID'] == i, 'AssignedID'] = station

gsa_df.replace(0, np.nan, inplace=True)
gsa_df.to_csv('geoglows_station_assigned.csv', index=False)



