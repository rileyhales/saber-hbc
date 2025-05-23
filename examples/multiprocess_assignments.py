import logging
import os
from multiprocessing import Pool

import pandas as pd

import saber
from saber.io import COL_ASN_GID
from saber.io import COL_ASN_MID
from saber.io import COL_ASN_REASON
from saber.io import COL_CID
from saber.io import COL_GID
from saber.io import COL_MID

logger = logging.getLogger(__name__)

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
                pd.concat(p.starmap(saber.assign._map_assign_ungauged, [(df, c_df, x) for x in mids])),
                df[~df[COL_MID].isin(mids)]
            ]).reset_index(drop=True)
            df.to_parquet(f'./assign_table_after_cluster_{cluster_number}.parquet')
