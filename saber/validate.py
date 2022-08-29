import glob
import os
import shutil

import netCDF4 as nc
import numpy as np
import pandas as pd

from .io import cal_nc_name
from .io import gid_col
from .io import metric_nc_name_list
from .io import mid_col
from .io import read_table
from .io import scaffold_workdir


def sample_gauges(workdir: str, overwrite: bool = False) -> None:
    """
    Prepares working directories in the validation_runs directory and populates the data_processed and gis_inputs
    folders with data from the master working directory. The gauges tables placed in the validation runs are randomly
    chosen samples of the master gauge table.

    Args:
        workdir: the project working directory
        overwrite: delete the old validation directory before sampling

    Returns:
        None
    """
    vr_path = os.path.join(workdir, 'validation')
    if overwrite:
        if os.path.exists(vr_path):
            shutil.rmtree(vr_path)
        os.mkdir(vr_path)

    gt = read_table(workdir, 'gauge_table')
    initial_row_count = gt.shape[0]

    rows_to_drop = round(gt.shape[0] / 10)

    for i in range(5):
        # some math related to which iteration of filtering we're on
        n = initial_row_count - rows_to_drop * (i + 1)
        pct_remain = 100 - 10 * (i + 1)
        subpath = os.path.join(vr_path, f'{pct_remain}')

        # create the new project working directory
        scaffold_workdir(subpath, include_validation=False)

        # overwrite the processed data directory so we don't need to redo this each time
        shutil.copytree(
            os.path.join(workdir, 'data_processed'),
            os.path.join(subpath, 'data_processed'),
            dirs_exist_ok=True
        )

        # sample the gauge table
        gt = gt.sample(n=n)
        gt.to_csv(os.path.join(subpath, 'gis', 'gauge_table.csv'), index=False)
        shutil.copyfile(os.path.join(workdir, 'gis', 'drain_table.csv'),
                        os.path.join(subpath, 'gis', 'drain_table.csv'))

        # filter the copied processed data to only contain the gauges included in this filtered step
        processed_sim_data = glob.glob(os.path.join(subpath, 'data_processed', 'obs-*.csv'))
        for f in processed_sim_data:
            a = pd.read_csv(f, index_col=0)
            a = a.filter(items=gt[gid_col].astype(str))
            a.to_csv(f)

    return


def gen_val_table(workdir: str) -> pd.DataFrame:
    """
    Prepares the validation summary table that contains the list of gauged rivers plus their statistics computed in
    each validation run. Used to create gis files for mapping the results.

    Args:
        workdir: the project working directory

    Returns:
        pandas.DataFrame
    """
    df = pd.read_csv(os.path.join(workdir, 'gis', 'gauge_table.csv'))
    df['100'] = 1

    stats_df = {}
    a = nc.Dataset(os.path.join(workdir, cal_nc_name))
    stats_df[mid_col] = np.asarray(a[mid_col][:])
    for metric in metric_nc_name_list:
        arr = np.asarray(a[metric][:])
        stats_df[f'{metric}_raw'] = arr[:, 0]
        stats_df[f'{metric}_adj'] = arr[:, 1]
    a.close()

    for d in sorted(
            [a for a in glob.glob(os.path.join(workdir, 'validation', '*')) if os.path.isdir(a)],
            reverse=True
    ):
        val_percent = os.path.basename(d)
        valset_gids = pd.read_csv(os.path.join(d, 'gis', 'gauge_table.csv'))[gid_col].values.tolist()

        # mark a column indicating the gauges included in the validation set
        df[val_percent] = 0
        df.loc[df[gid_col].isin(valset_gids), val_percent] = 1

        # add columns for the metrics of all gauges during this validation set
        a = nc.Dataset(os.path.join(d, cal_nc_name))
        for metric in metric_nc_name_list:
            stats_df[f'{metric}_{val_percent}'] = np.asarray(a[metric][:])[:, 1]
        a.close()

    # merge gauge_table with the stats, save and return
    df = df.merge(pd.DataFrame(stats_df), on=mid_col, how='inner')
    df.to_csv(os.path.join(workdir, 'validation', 'val_table.csv'), index=False)

    return df
