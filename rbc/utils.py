import pandas as pd
import geopandas as gpd


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
