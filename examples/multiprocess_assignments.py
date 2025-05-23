import logging
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from saber.io import COL_ASN_GID
from saber.io import COL_ASN_MID
from saber.io import COL_ASN_REASON
from saber.io import COL_CID
from saber.io import COL_GID
from saber.io import COL_GPROP
from saber.io import COL_MID
from saber.io import COL_RID
from saber.io import COL_RPROP
from saber.io import COL_X
from saber.io import COL_Y

logger = logging.getLogger(__name__)


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
        # todo check if there is a closer gauge *between* the stream and the reg structure
        if new_row[COL_RPROP].values[0] != '' or new_row[COL_RID].values[0] is not None:
            if new_row[COL_RPROP].values[0]:
                potential_mid = new_row[COL_RPROP].values[0].split('-')[-1]  # Find the MID of the reg structure
            else:
                potential_mid = new_row[COL_MID].values[0]  # use current row because it has the reg structure
            potential_gid = assign_df[assign_df[COL_MID] == potential_mid][COL_GID].values[0]
            if potential_gid != '':
                new_row[COL_ASN_MID] = potential_mid
                new_row[COL_ASN_GID] = potential_gid
                new_row[COL_ASN_REASON] = 'regulatory'
                return new_row

        # if the stream is near a gauge, assign that gauge
        if new_row[COL_GPROP].values[0] != '':
            new_row[COL_ASN_MID] = new_row[COL_GPROP].values[0].split('-')[-1]
            new_row[COL_ASN_GID] = assign_df[assign_df[COL_MID] == new_row[COL_ASN_MID].values[0]][COL_GID].values[0]
            new_row[COL_ASN_REASON] = 'near_gauge'
            return new_row

        # find the closest gauge (no accounting for projection/map distortion)
        mid_x, mid_y = assign_df.loc[assign_df[COL_MID] == mid, [COL_X, COL_Y]].head(1).values.flatten()
        cluster_num = new_row[COL_CID].values[0]
        cluster_filter = gauges_df[COL_CID] == cluster_num
        if np.nansum(cluster_filter) > 0:
            gauges_df = gauges_df[cluster_filter]
        row_idx_to_assign = pd.Series(np.sqrt(
            ((gauges_df[COL_X] - mid_x) ** 2) + ((gauges_df[COL_Y] - mid_y) ** 2)
        )).idxmin()
        asgn_mid, asgn_gid = gauges_df.loc[row_idx_to_assign, [COL_MID, COL_GID]]
        asgn_reason = f'nearest_cluster_{cluster_num}'
        new_row[[COL_ASN_MID, COL_ASN_GID, COL_ASN_REASON]] = [asgn_mid, asgn_gid, asgn_reason]
        return new_row

    except Exception as e:
        logger.error(f'Error (mid: {mid}): {e}')
        return assign_df.head(0)


if __name__ == '__main__':
    df = pd.read_parquet('./assign_table.parquet')
    selector = df[COL_GID].notna()
    df.loc[selector, COL_ASN_MID] = df[COL_MID]
    df.loc[selector, COL_ASN_GID] = df[COL_GID]
    df.loc[selector, COL_ASN_REASON] = 'gauged'

    with Pool(os.cpu_count()) as p:
        logger.info('Assign Basins within Clusters')
        for cluster_number in range(df[COL_CID].max() + 1):
            logger.info(f'Assigning basins in cluster {cluster_number}')
            # filter assign dataframe to only gauged basins within the cluster
            c_df = df[df[COL_CID] == cluster_number]
            c_df = c_df[c_df[COL_GID].notna()]
            # keep a list of the unassigned basins in the cluster
            mids = c_df[c_df[COL_ASN_REASON] == 'unassigned'][COL_MID].values
            df = pd.concat([
                pd.concat(p.starmap(_map_assign_ungauged, [(df, c_df, x) for x in mids])),
                df[~df[COL_MID].isin(mids)]
            ]).reset_index(drop=True)
            df.to_parquet(f'./assign_table_after_cluster_{cluster_number}.parquet')
