import logging
import os
from glob import glob
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

import saber
from saber.io import COL_ASN_GID, COL_ASN_MID, COL_ASN_REASON, COL_CID, COL_GID, COL_MID, COL_GPROP, COL_RPROP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global variables for worker processes
_global_cdf = None
_global_gdf = None


def init_globals(cluster_df):
    global _global_cdf
    _global_cdf = cluster_df


def assign_wrapper(mid):
    return saber.assign._map_assign_ungauged(_global_cdf, _global_gdf, mid)


if __name__ == '__main__':
    gauge_data_directory = ''
    df = pd.read_parquet('./assign_table-init.parquet')
    df[COL_GPROP] = ''
    df[COL_RPROP] = ''
    df[COL_ASN_GID] = ''
    df[COL_ASN_MID] = ''
    df[COL_ASN_REASON] = 'unassigned'
    df = df.reset_index(drop=True)
    df.loc[df[COL_GID].notna(), COL_ASN_MID] = df[COL_MID]
    df.loc[df[COL_GID].notna(), COL_ASN_GID] = df[COL_GID]
    df.loc[df[COL_GID].notna(), COL_ASN_REASON] = 'gauged'

    # verify that all gauges have a csv
    gauges = df[df[COL_GID].notna()][COL_GID].unique()
    gauges_exist = [os.path.exists(os.path.join(gauge_data_directory, f'{gid}.csv')) for gid in gauges]
    if not all(gauges_exist):
        df.loc[df[COL_GID].isin(gauges[~pd.Series(gauges_exist)]), COL_GID] = None
    df.to_parquet('./assign_table-validated.parquet')

    df = saber.table.mp_prop_gauges(df)
    df.to_parquet('./assign_table-gaugeprop.parquet')
    df = saber.table.mp_prop_regulated(df)
    df.to_parquet('./assign_table-regprop.parquet')

    df = pd.read_parquet('./assign_table-regprop.parquet')
    df.reset_index(drop=True, inplace=True)
    df.loc[df[COL_GPROP].ne(''), COL_ASN_REASON] = 'gauge_prop'
    df.loc[df[COL_GPROP].ne(''), COL_ASN_MID] = df[COL_GPROP].str.split('-').str[-1]
    df.loc[df[COL_RPROP].ne(''), COL_ASN_REASON] = 'reg_prop'
    df.loc[df[COL_RPROP].ne(''), COL_ASN_MID] = df[COL_RPROP].str.split('-').str[-1]
    df['asgn_gid'] = df['asgn_mid'].map(dict(zip(df['model_id'], df['gauge_id'])))
    df.to_parquet('./assign_table-resolvedprop.parquet')

    _global_gdf = df[df[COL_GID].notna()].copy(deep=True)
    print('Assign Basins within Clusters')
    for cluster_number in range(df[COL_CID].max() + 1):
        table_name = f'assign_table-cluster{cluster_number}.parquet'
        if os.path.exists(table_name):
            print(f'Skipping cluster {cluster_number}, already processed.')
            continue
        c_df = df[df[COL_CID] == cluster_number]
        mids = c_df[c_df[COL_ASN_REASON] == 'unassigned'][COL_MID].values
        print(f'Assigning basins in cluster {cluster_number}')
        print(mids.shape)

        with Pool(os.cpu_count(), initializer=init_globals, initargs=(c_df,)) as p:
            results = list(tqdm(
                p.imap_unordered(assign_wrapper, mids, chunksize=50),
                total=len(mids),
                desc=f'Cluster {cluster_number}'
            ))

        pd.concat(results).reset_index(drop=True).to_parquet(table_name)

    cluster_tables = pd.concat([pd.read_parquet(f) for f in glob('./assign_table-cluster*.parquet')], ignore_index=True)
    prop_assigned = df[df[COL_ASN_REASON].ne('unassigned')].reset_index(drop=True)
    pd.concat([prop_assigned, cluster_tables]).reset_index(drop=True).to_parquet('./assign_table-final.parquet')
