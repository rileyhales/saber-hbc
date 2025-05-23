import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import saber
from saber.io import COL_ASN_GID
from saber.io import COL_ASN_MID
from saber.io import COL_ASN_REASON
from saber.io import COL_CID
from saber.io import COL_GID
from saber.io import COL_MID

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def assign_ungauged_wrapper(mid, df, c_df):
    return saber.assign._map_assign_ungauged(df, c_df, mid)


if __name__ == '__main__':
    df = pd.read_parquet('./assign_table.parquet')

    # Initial assignments for gauged basins
    df.loc[df[COL_GID].notna(), COL_ASN_MID] = df[COL_MID]
    df.loc[df[COL_GID].notna(), COL_ASN_GID] = df[COL_GID]
    df.loc[df[COL_GID].notna(), COL_ASN_REASON] = 'gauged'

    # max_workers = os.cpu_count() * 2
    max_workers = os.cpu_count()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for cluster_number in range(df[COL_CID].max() + 1):
            logger.info(f'Assigning basins in cluster {cluster_number}')
            # Subset data for this cluster
            c_df = df[df[COL_CID] == cluster_number]
            mids = c_df[c_df[COL_ASN_REASON] == 'unassigned'][COL_MID].values
            print(mids)
            # Threaded assignment
            futures = {executor.submit(assign_ungauged_wrapper, mid, df, c_df): mid for mid in mids}
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error processing MID {futures[future]}: {e}")

            # Flatten the results and update the DataFrame
            if results:
                updated = pd.concat(results)
                df = pd.concat([
                    updated,
                    df[~df[COL_MID].isin(updated[COL_MID])]
                ]).reset_index(drop=True)

            df.to_parquet(f'./assign_table_after_cluster_{cluster_number}.parquet')
