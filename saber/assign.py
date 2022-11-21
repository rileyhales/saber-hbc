import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .io import asn_gid_col
from .io import asn_mid_col
from .io import cls_col
from .io import gid_col
from .io import mid_col
from .io import g_prop_col
from .io import reason_col
from .io import x_col
from .io import y_col

__all__ = ['assign_gauged', 'assign_propagation', 'mp_assign_clusters', ]

logger = logging.getLogger(__name__)


def mp_assign(df: pd.DataFrame, n_processes: int or None = None) -> pd.DataFrame:
    """
    Assigns basins a gauge for correction which contain a gauge

    Args:
        df: the assignments table dataframe
        n_processes: number of processes to use for multiprocessing, passed to Pool

    Returns:
        Copy of df1 with assignments made
    """
    df = assign_gauged(df)
    df = assign_propagation(df)
    df = mp_assign_clusters(df, n_processes)
    return df


def assign_gauged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns basins a gauge for correction which contain a gauge

    Args:
        df: the assignments table dataframe

    Returns:
        Copy of df1 with assignments made
    """
    selector = df[gid_col].notna()
    df.loc[selector, asn_mid_col] = df[mid_col]
    df.loc[selector, asn_gid_col] = df[gid_col]
    df.loc[selector, reason_col] = 'gauged'
    return df


def assign_propagation(df: pd.DataFrame) -> pd.DataFrame:
    """
     Merge the assignment table and the propagation assignments

    Args:
        df: the assignments table dataframe

    Returns:
        pd.DataFrame
    """
    # todo
    not_gauged = df[df[gid_col].isna()]

    # need to write values to the asgn mid and gid columns and also the reasons
    return


def mp_assign_clusters(df: pd.DataFrame, n_processes: int or None = None) -> pd.DataFrame:
    """

    Args:
        df: the assignments table dataframe with the clustering labels already applied
        n_processes: number of processes to use for multiprocessing, passed to Pool

    Returns:
        pd.DataFrame
    """
    # todo filter the dataframe
    with Pool(n_processes) as p:
        logger.info('Assign Basins within Clusters')
        for cluster_number in range(df[cls_col].max() + 1):
            logger.info(f'Assigning basins in cluster {cluster_number}')
            # limit by cluster number
            c_df = df[df[cls_col] == cluster_number]
            # keep a list of the unassigned basins in the cluster
            mids = c_df[c_df[reason_col] == 'unassigned'][mid_col].values
            # filter cluster dataframe to find only gauged basins
            c_df = c_df[c_df[gid_col].notna()]
            df = pd.concat([
                pd.concat(p.starmap(_map_assign_ungauged, [(df, c_df, x) for x in mids])),
                df[~df[mid_col].isin(mids)]
            ]).reset_index(drop=True)

    return df


def _map_assign_ungauged(assign_df: pd.DataFrame, gauges_df: np.array, mid: str) -> pd.DataFrame:
    """
    Assigns all possible ungauged basins a gauge that is
        (1) is closer than any other gauge
        (2) is of same stream order as ungauged basin
        (3) in the same simulated fdc cluster as ungauged basin

    Args:
        assign_df: the assignments table dataframe
        gauges_df: a subset of the assignments dataframe containing the gauges
        mid: the model_id to assign a gauge for

    Returns:
        a new row for the given mid with the assignments made
    """
    try:
        new_row = assign_df[assign_df[mid_col] == mid].copy()
        if new_row[g_prop_col].values[0] != '':
            prop_str = new_row[g_prop_col].values[0]
            asgn_mid = prop_str.split('-')[-1]
            asgn_gid = assign_df[assign_df[mid_col] == asgn_mid][asn_gid_col].values[0]
            asgn_reason = prop_str

        else:
            # find the closest gauge using euclidean distance without accounting for projection/map distortion
            mid_x, mid_y = assign_df.loc[assign_df[mid_col] == mid, [x_col, y_col]].head(1).values.flatten()
            row_idx_to_assign = pd.Series(
                np.sqrt(np.power(gauges_df[x_col] - mid_x, 2) + np.power(gauges_df[y_col] - mid_y, 2))
            ).idxmin()
            asgn_mid, asgn_gid = gauges_df.loc[row_idx_to_assign, [mid_col, gid_col]]
            asgn_reason = f'cluster-{gauges_df[cls_col].values[0]}'

        new_row[[asn_mid_col, asn_gid_col, reason_col]] = [asgn_mid, asgn_gid, asgn_reason]
    except Exception as e:
        logger.error(f'Error in map_assign_ungauged: {e}')
        new_row = pd.DataFrame(columns=assign_df.columns)

    return new_row
