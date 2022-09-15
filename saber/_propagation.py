import datetime
import logging
from multiprocessing import Pool

import pandas as pd

from saber.io import mid_col, gid_col, reason_col, order_col, down_mid_col, asgn_mid_col, asgn_gid_col

logger = logging.getLogger(__name__)


def map_assign_down(start_mid: int) -> pd.DataFrame:
    return map_assign_propagation(start_mid, 'downstream')


def map_assign_up(start_mid: int) -> pd.DataFrame:
    return map_assign_propagation(start_mid, 'upstream')


def map_assign_propagation(start_mid: int, direction: str) -> pd.DataFrame:
    df = pd.read_parquet('assign_table_slim.parquet', engine='fastparquet')
    start_id_order = df[df[mid_col] == start_mid][order_col].values[0]
    df = df[df[order_col] == start_id_order]

    stream_row = df[df[mid_col] == start_mid]
    start_gid = stream_row[asgn_gid_col].values[0]

    n_steps = 1

    while len(stream_row):

        df.loc[df[mid_col] == stream_row[down_mid_col].values[0], [asgn_mid_col, asgn_gid_col, reason_col]] = \
            [start_mid, start_gid, f'propagation-downstream-{n_steps}']

        if direction == 'downstream':
            stream_row = df[df[mid_col] == stream_row[down_mid_col].values[0]]
        elif direction == 'upstream':
            stream_row = df[df[down_mid_col] == stream_row[mid_col].values[0]]
        else:
            raise ValueError(f'Direction should be "upstream" or "downstream", not {direction}')

        n_steps += 1

        # repeat while the next downstream is not -1 (outlet)
        if len(stream_row) == 0 or stream_row[down_mid_col].values[0] == -1:
            break
    return df[df[asgn_mid_col] == start_mid]


# if __name__ == '__main__':
#     df = pd.read_parquet('assign_table_slim.parquet', engine='fastparquet')
#     df = df.sort_values(by=mid_col)
#
#     # todo use starmap to pass pointer to df, only copies sections if needed to reduce memory and read times
#     # todo test mapping speeds
#     # todo test memory usage
#     # todo documentation
#
#     gauged_mids = df.loc[df[gid_col].notna(), mid_col]
#
#     t1 = datetime.datetime.now()
#     with Pool(18) as p:
#         df_prop_down = pd.concat(p.map(map_assign_down, gauged_mids))
#         df_prop_up = pd.concat(p.map(map_assign_up, gauged_mids))
#     t2 = datetime.datetime.now()
#     print(f'Parallel processing took {(t2 - t1).total_seconds() / 60} minutes')
#
#     df_prop_down = df_prop_down.sort_values(by=mid_col)
#     df_prop_up = df_prop_up.sort_values(by=mid_col)
#
#     df_prop_down.to_parquet('assign_table_prop_down.parquet', engine='fastparquet')
#     df_prop_up.to_parquet('assign_table_prop_up.parquet', engine='fastparquet')
#
#     df.loc[df[mid_col].isin(df_prop_up[mid_col]), [asgn_mid_col, asgn_gid_col, reason_col]] = \
#         df_prop_up[[asgn_mid_col, asgn_gid_col, reason_col]]
#
#     df.loc[df[mid_col].isin(df_prop_down[mid_col]), [asgn_mid_col, asgn_gid_col, reason_col]] = \
#         df_prop_down[[asgn_mid_col, asgn_gid_col, reason_col]]
