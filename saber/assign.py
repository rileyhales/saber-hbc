import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .io import COL_ASN_GID
from .io import COL_ASN_MID
from .io import COL_ASN_REASON
from .io import COL_CID
from .io import COL_GPROP
from .io import COL_GID
from .io import COL_MID
from .io import COL_X
from .io import COL_Y
from .io import get_state
from .io import read_table

__all__ = ['mp_assign', 'assign_gauged', 'assign_propagation', 'mp_assign_clusters', ]

logger = logging.getLogger(__name__)


def mp_assign(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Assigns basins a gauge for correction which contain a gauge

    Args:
        df: the assignments table dataframe

    Returns:
        Copy of df1 with assignments made
    """
    if df is None:
        df = read_table('assign_table')
    df = assign_gauged(df)
    df = assign_propagation(df)
    df = mp_assign_clusters(df, get_state('n_processes'))
    return df


def assign_gauged(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns basins a gauge for correction which contain a gauge

    Args:
        df: the assignments table dataframe

    Returns:
        Copy of df1 with assignments made
    """
    selector = df[COL_GID].notna()
    df.loc[selector, COL_ASN_MID] = df[COL_MID]
    df.loc[selector, COL_ASN_GID] = df[COL_GID]
    df.loc[selector, COL_ASN_REASON] = 'gauged'
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
    not_gauged = df[df[COL_GID].isna()]

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
        for cluster_number in range(df[COL_CID].max() + 1):
            logger.info(f'Assigning basins in cluster {cluster_number}')
            # limit by cluster number
            c_df = df[df[COL_CID] == cluster_number]
            # keep a list of the unassigned basins in the cluster
            mids = c_df[c_df[COL_ASN_REASON] == 'unassigned'][COL_MID].values
            # filter cluster dataframe to find only gauged basins
            c_df = c_df[c_df[COL_GID].notna()]
            df = pd.concat([
                pd.concat(p.starmap(_map_assign_ungauged, [(df, c_df, x) for x in mids])),
                df[~df[COL_MID].isin(mids)]
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
        new_row = assign_df[assign_df[COL_MID] == mid].copy()
        if new_row[COL_GPROP].values[0] != '':
            prop_str = new_row[COL_GPROP].values[0]
            asgn_mid = prop_str.split('-')[-1]
            asgn_gid = assign_df[assign_df[COL_MID] == asgn_mid][COL_ASN_GID].values[0]
            asgn_reason = prop_str

        else:
            # find the closest gauge using euclidean distance without accounting for projection/map distortion
            mid_x, mid_y = assign_df.loc[assign_df[COL_MID] == mid, [COL_X, COL_Y]].head(1).values.flatten()
            row_idx_to_assign = pd.Series(
                np.sqrt(np.power(gauges_df[COL_X] - mid_x, 2) + np.power(gauges_df[COL_Y] - mid_y, 2))
            ).idxmin()
            asgn_mid, asgn_gid = gauges_df.loc[row_idx_to_assign, [COL_MID, COL_GID]]
            asgn_reason = f'cluster-{gauges_df[COL_CID].values[0]}'

        new_row[[COL_ASN_MID, COL_ASN_GID, COL_ASN_REASON]] = [asgn_mid, asgn_gid, asgn_reason]
    except Exception as e:
        logger.error(f'Error in map_assign_ungauged: {e}')
        new_row = pd.DataFrame(columns=assign_df.columns)

    return new_row
