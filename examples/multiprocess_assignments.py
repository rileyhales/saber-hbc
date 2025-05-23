import logging
import os
from multiprocessing import Pool

import pandas as pd

import saber
from saber.io import COL_ASN_GID, COL_ASN_MID, COL_ASN_REASON, COL_CID, COL_GID, COL_MID

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global variables for worker processes
_global_df = None
_global_cdf = None


def init_globals(df, c_df):
    global _global_df, _global_cdf
    _global_df = df
    _global_cdf = c_df


def assign_wrapper(mid):
    return saber.assign._map_assign_ungauged(_global_df, _global_cdf, mid)


df = pd.read_parquet('./assign_table.parquet')

if __name__ == '__main__':
    selector = df[COL_GID].notna()
    df.loc[selector, COL_ASN_MID] = df[COL_MID]
    df.loc[selector, COL_ASN_GID] = df[COL_GID]
    df.loc[selector, COL_ASN_REASON] = 'gauged'

    logger.info('Assign Basins within Clusters')

    for cluster_number in range(df[COL_CID].max() + 1):
        logger.info(f'Assigning basins in cluster {cluster_number}')
        c_df = df[df[COL_CID] == cluster_number]
        mids = c_df[c_df[COL_ASN_REASON] == 'unassigned'][COL_MID].values

        with Pool(os.cpu_count(), initializer=init_globals, initargs=(df, c_df)) as p:
            results = p.map(assign_wrapper, mids)

        updated = pd.concat(results)
        df = pd.concat([
            updated,
            df[~df[COL_MID].isin(updated[COL_MID])]
        ]).reset_index(drop=True)

        df.to_parquet(f'./assign_table_after_cluster_{cluster_number}.parquet')
