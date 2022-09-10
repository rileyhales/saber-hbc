import logging

import pandas as pd

from .io import asgn_gid_col
from .io import asgn_mid_col
from .io import down_mid_col
from .io import mid_col
from .io import order_col
from .io import reason_col

logger = logging.getLogger(__name__)


def walk_downstream(df: pd.DataFrame, start_id: int, same_order: bool = True, outlet_next_id: str or int = -1) -> tuple:
    """
    Traverse a stream network table containing a column of unique ids, a column of the id for the stream/basin
    downstream of that point, and, optionally, a column containing the stream order.

    Args:
        df (pd.DataFrame):
        start_id (int):
        same_order (bool):
        outlet_next_id (str or int):

    Returns:
        Tuple of stream ids in the order they come from the starting point.
    """
    downstream_ids = []

    df_ = df.copy()
    if same_order:
        start_id_order = df_[df_[mid_col] == start_id][order_col].values[0]
        df_ = df_[df_[order_col] == start_id_order]

    stream_row = df_[df_[mid_col] == start_id]
    while len(stream_row[down_mid_col].values) > 0 and stream_row[down_mid_col].values[0] != outlet_next_id:
        downstream_ids.append(stream_row[down_mid_col].values[0])
        stream_row = df_[df_[mid_col] == stream_row[down_mid_col].values[0]]
        if len(stream_row) == 0:
            break
    return tuple(downstream_ids)


def walk_upstream(df: pd.DataFrame, start_id: int, same_order: bool = True) -> tuple:
    """
    Traverse a stream network table containing a column of unique ids, a column of the id for the stream/basin
    downstream of that point, and, optionally, a column containing the stream order.

    Args:
        df (pd.DataFrame): the assign_table df
        start_id (int): that id to start on
        same_order (bool): select upstream segments only on the same order stream

    Returns:
        Tuple of stream ids in the order they come from the starting point. If you chose same_order = False, the
        streams will appear in order on each upstream branch but the various branches will appear mixed in the tuple in
        the order they were encountered by the iterations.
    """
    df_ = df.copy()
    if same_order:
        df_ = df_[df_[order_col] == df_[df_[mid_col] == start_id][order_col].values[0]]

    # start a list of the upstream ids
    upstream_ids = [start_id, ]
    upstream_rows = df_[df_[down_mid_col] == start_id]

    while not upstream_rows.empty or len(upstream_rows) > 0:
        if len(upstream_rows) == 1:
            upstream_ids.append(upstream_rows[mid_col].values[0])
            upstream_rows = df_[df_[down_mid_col] == upstream_rows[mid_col].values[0]]
        elif len(upstream_rows) > 1:
            for id in upstream_rows[mid_col].values.tolist():
                upstream_ids += list(walk_upstream(df_, id, False))
                upstream_rows = df_[df_[down_mid_col] == id]
    return tuple(set(upstream_ids))


def propagate_in_table(df: pd.DataFrame, start_mid: int, start_gid: int, connected: tuple, max_prop: int,
                       direction: str):
    """

    Args:
        df: the assign_table df
        start_mid: the model_id of the stream to start at
        connected: a list of stream segments, up- or downstream from the gauged_stream, in the order they come
        max_prop: max number of stream segments to propagate up/downstream
        direction: either "upstream" or "downstream"

    Returns:
        df
    """
    _df = df.copy()

    for index, segment_id in enumerate(connected):
        distance = index + 1
        if distance > max_prop:
            continue
        downstream_row = _df[_df[mid_col] == segment_id]

        # if the downstream segment doesn't have an assigned gauge, we're going to assign the current one
        if downstream_row[asgn_mid_col].empty or pd.isna(downstream_row[asgn_mid_col]).any():
            _df.loc[_df[mid_col] == segment_id, [asgn_mid_col, asgn_gid_col, reason_col]] = \
                [start_mid, start_gid, f'propagation-{direction}-{distance}']
            logger.info(f'assigned gauged stream {start_mid} to ungauged {direction} {segment_id}')
            continue

        # if the stream segment does have an assigned value, check the value to determine what to do
        else:
            downstream_reason = downstream_row[reason_col].values[0]
            if downstream_reason == 'gauged':
                logger.info('already gauged')
            if 'propagation' in downstream_reason and int(str(downstream_reason).split('-')[-1]) >= distance:
                _df.loc[_df[mid_col] == segment_id, [asgn_mid_col, asgn_gid_col, reason_col]] = \
                    [start_mid, start_gid, f'propagation-{direction}-{distance}']
                logger.info(f'assigned gauged stream {start_mid} to previously assigned {direction} {segment_id}')
    return _df
