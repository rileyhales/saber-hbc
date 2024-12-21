import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .io import COL_ASN_GID
from .io import COL_ASN_MID
from .io import COL_ASN_REASON
from .io import COL_CID
from .io import COL_GID
from .io import COL_GPROP
from .io import COL_MID
from .io import COL_RID
from .io import COL_RPROP
from .io import COL_X
from .io import COL_Y
from .io import get_state
from .io import read_table

_all_ = ['mp_assign', 'assign_gauged', 'mp_assign_ungauged', ]

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
    df = mp_assign_ungauged(df)
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
    df.reset_index(drop=True, inplace=True)
    selector.reset_index(drop=True, inplace=True)
  #  print(df[COL_GID])
    df.loc[selector, COL_ASN_MID] = df[COL_MID] # true aako thau ma asgn_mid column ma model_id rakcha
    df.loc[selector, COL_ASN_GID] = df[COL_GID] # true aako thau ma asgn_gid column ma gauge_id rakcha
    df.loc[selector, COL_ASN_REASON] = 'gauged'
    return df


def mp_assign_ungauged(df: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        df: the assignments table dataframe with the clustering labels already applied

    Returns:
        pd.DataFrame
    """
    with Pool(get_state('n_processes')) as p:
        logger.info('Assign Basins within Clusters')
        for cluster_number in range(df[COL_CID].max() + 1):
            logger.info(f'Assigning basins in cluster {cluster_number}')
            # filter assign dataframe to only gauged basins within the cluster
            gauge_clstr = pd.read_csv('/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/gauge_table_2nd_iteration_deDuplicated_cleaned_with_climate.csv')
            df = df.merge(gauge_clstr[['File', 'clst_gauge','Climate']], left_on='gauge_id', right_on='File', how='inner')
            c_df = df[df[COL_CID] == cluster_number]
            # keep a list of the unassigned basins in the cluster
            mids = c_df[c_df[COL_ASN_REASON] == 'unassigned'][COL_MID].values
            df = pd.concat([
                pd.concat(p.starmap(_map_assign_ungauged, [(df, c_df, x) for x in mids])),
                df[~df[COL_MID].isin(mids)]
            ]).reset_index(drop=True)

    return df


def _map_assign_ungauged(assign_df: pd.DataFrame, gauges_df: pd.DataFrame, mid: str) -> pd.DataFrame:
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
        if isinstance(mid, pd.Series):
            mid = mid.values[0]
        new_row = assign_df[assign_df[COL_MID] == mid].copy()

        # if the stream contains or is downstream of a regulatory structure check is reg structure contains a gauge
        # check is separate from gauge prop, so it are assigned even during bootstrapping
        # todo check if there is a closer gauge between the stream and the reg structure

        #regid, rprop and gprop exist in assign_df then
        if (new_row[COL_GPROP].values[0] != ''
                and new_row[COL_RPROP].values[0] != ''
                and new_row[COL_RID].values[0] is not None):
            # Extract up or down and the number
            direction_R = new_row[COL_RPROP].values[0].split('-')[0]
            direction_R_n = int(new_row[COL_RPROP].values[0].split('-')[1])
            direction_G = new_row[COL_GPROP].values[0].split('-')[0]
            direction_G_n = int(new_row[COL_GPROP].values[0].split('-')[1])

            # if gprop is up or rprop down with less number then use spatial analysis
            if (direction_R == 'down' and
                    (direction_G == 'up' or (direction_G == 'down' and direction_R_n < direction_G_n))):
                return assign_nearest_cluster(new_row, assign_df, gauges_df, mid).assign(**{COL_ASN_REASON: 'regulatory'})
            else:  # if gprop is down and its number is equal to or less, extract the value
                potential_mid = new_row[COL_GPROP].values[0].split('-')[-1]
                potential_gid = assign_df[assign_df[COL_MID] == potential_mid][COL_GID].values[0]
                asgn_reason = 'near_gauge'
                new_row[[COL_ASN_MID, COL_ASN_GID, COL_ASN_REASON]] = [potential_mid, potential_gid, asgn_reason]
                return new_row

        # if only regid and gprop are there in assign table
        elif new_row[COL_GPROP].values[0] != '' and new_row[COL_RID].values[0] is not None and new_row[COL_RPROP].values[0] == '':
            direction_G = new_row[COL_GPROP].values[0].split('-')[0]
            # if there is a gauge in down direction then use it otherwise use spatial analysis
            if direction_G == 'down':
                potential_mid = new_row[COL_GPROP].values[0].split('-')[-1]
                potential_gid = assign_df[assign_df[COL_MID] == potential_mid][COL_GID].values[0]
                asgn_reason = 'near_gauge'
                new_row[[COL_ASN_MID, COL_ASN_GID, COL_ASN_REASON]] = [potential_mid, potential_gid, asgn_reason]
                return new_row
            else:  # if up then we can't use that gauge so do spatial analysis
                return assign_nearest_cluster(new_row, assign_df, gauges_df, mid).assign(**{COL_ASN_REASON: 'regulatory'})

        # rprop and gprop are there
        elif new_row[COL_RPROP].values[0] != '' and new_row[COL_GPROP].values[0] != '' and new_row[COL_RID].values[
            0] is None:
            direction_R = new_row[COL_RPROP].values[0].split('-')[0]
            direction_R_n = int(new_row[COL_RPROP].values[0].split('-')[1])
            direction_G = new_row[COL_GPROP].values[0].split('-')[0]
            direction_G_n = int(new_row[COL_GPROP].values[0].split('-')[1])

            # Without regulated structure if the gprop is down with number less use it, if gprop is up use that
            if direction_R == 'down' and (direction_G == 'down' and direction_R_n < direction_G_n):
                return assign_nearest_cluster(new_row, assign_df, gauges_df, mid).assign(**{COL_ASN_REASON: 'regulatory'})
            else:
                potential_mid = new_row[COL_GPROP].values[0].split('-')[-1]
                potential_gid = assign_df[assign_df[COL_MID] == potential_mid][COL_GID].values[0]
                asgn_reason = 'near_gauge'
                new_row[[COL_ASN_MID, COL_ASN_GID, COL_ASN_REASON]] = [potential_mid, potential_gid, asgn_reason]
                return new_row

        # only gprop use the gprop
        elif new_row[COL_GPROP].values[0] != '':
            new_row[COL_ASN_MID] = new_row[COL_GPROP].values[0].split('-')[-1]
            new_row[COL_ASN_GID] = assign_df[assign_df[COL_MID] == new_row[COL_ASN_MID].values[0]][COL_GID].values[0]
            new_row[COL_ASN_REASON] = 'near_gauge'
            return new_row

        else:  # use spatial analysis
            return assign_nearest_cluster(new_row, assign_df, gauges_df, mid)

    except Exception as e:
        logger.error(f'Error (mid: {mid}): {e}')
        return assign_df.head(0)

def assign_nearest_cluster(new_row, assign_df, gauges_df, mid):
    # Extract necessary values
    mid_x, mid_y, strahler_order = assign_df.loc[
        assign_df[COL_MID] == mid, [COL_X, COL_Y, 'strahler_order']].head(1).values.flatten()
    cluster_num = new_row[COL_CID].values[0]
    gauge_clstr_num = new_row['clstr_gauge'].values[0]
    climate = new_row['Climate'].values[0]
    cluster_filter = ((gauges_df[COL_CID] == cluster_num)
                      # & (
                          #gauges_df['strahler_order'].between(strahler_order - 1, strahler_order + 1))
                          #gauges_df['strahler_order'].between(strahler_order - 1, strahler_order + 1) & (gauges_df['clstr_gauge'] == gauge_clstr_num) & (gauges_df['Climate'] == climate))
                          (gauges_df['clstr_gauge'] == gauge_clstr_num) & (gauges_df['Climate'] == climate))
    # Filter gauges_df if applicable
    if np.nansum(cluster_filter) > 0:
        gauges_df = gauges_df[cluster_filter]

    # Find the nearest gauge
    row_idx_to_assign = pd.Series(
        np.sqrt(((gauges_df[COL_X] - mid_x) ** 2) + ((gauges_df[COL_Y] - mid_y) ** 2))).idxmin()
    asgn_mid, asgn_gid = gauges_df.loc[row_idx_to_assign, [COL_MID, COL_GID]]
    asgn_reason = f'nearest_cluster_{cluster_num}'
    # Assign the values to new_row
    new_row[[COL_ASN_MID, COL_ASN_GID, COL_ASN_REASON]] = [asgn_mid, asgn_gid, asgn_reason]

    return new_row


#         if new_row[COL_RPROP].values[0] != '' or new_row[COL_RID].values[0] is not None:
#             if new_row[COL_RPROP].values[0]:
#                 potential_mid = new_row[COL_RPROP].values[0].split('-')[-1]  # Find the MID of the reg structure
#             else:
#                 potential_mid = new_row[COL_MID].values[0]  # use current row because it has the reg structure
#             potential_gid = assign_df[assign_df[COL_MID] == potential_mid][COL_GID].values[0]
#             if potential_gid != '':
#                 new_row[COL_ASN_MID] = potential_mid
#                 new_row[COL_ASN_GID] = potential_gid
#                 new_row[COL_ASN_REASON] = 'regulatory'
#                 return new_row
#
#         # if the stream is near a gauge, assign that gauge
#         if new_row[COL_GPROP].values[0] != '':
#             new_row[COL_ASN_MID] = new_row[COL_GPROP].values[0].split('-')[-1]
#             new_row[COL_ASN_GID] = assign_df[assign_df[COL_MID] == new_row[COL_ASN_MID].values[0]][COL_GID].values[0]
#             new_row[COL_ASN_REASON] = 'near_gauge'
#             return new_row
#
#         # find the closest gauge (no accounting for projection/map distortion)
#         mid_x, mid_y = assign_df.loc[assign_df[COL_MID] == mid, [COL_X, COL_Y]].head(1).values.flatten()
#         cluster_num = new_row[COL_CID].values[0]
#         cluster_filter = gauges_df[COL_CID] == cluster_num
#         if np.nansum(cluster_filter) > 0:
#             gauges_df = gauges_df[cluster_filter]
#         row_idx_to_assign = pd.Series(np.sqrt(
#             ((gauges_df[COL_X] - mid_x) ** 2) + ((gauges_df[COL_Y] - mid_y) ** 2)
#         )).idxmin()
#         asgn_mid, asgn_gid = gauges_df.loc[row_idx_to_assign, [COL_MID, COL_GID]]
#         asgn_reason = f'nearest_cluster_{cluster_num}'
#         new_row[[COL_ASN_MID, COL_ASN_GID, COL_ASN_REASON]] = [asgn_mid, asgn_gid, asgn_reason]
#         return new_row
#
#     except Exception as e:
#         logger.error(f'Error (mid: {mid}): {e}')
#         return assign_df.head(0)