import glob
import os
import shutil

import pandas as pd

from .prep import scaffold_workdir
from ._workflow import analyze_region


def sample_gauges(workdir: str) -> None:
    vr_path = os.path.join(workdir, 'validation_runs')
    if os.path.exists(vr_path):
        shutil.rmtree(vr_path)
    os.mkdir(vr_path)

    gt = pd.read_csv(os.path.join(workdir, 'gis_inputs', 'gauge_table.csv'))
    start_row_count = gt.shape[0]
    rows_to_drop = round(gt.shape[0] / 10)

    for i in range(5):
        # some math related to which iteration of filtering we're on
        n = start_row_count - rows_to_drop * (i + 1)
        pct_remain = 100 - 10 * (i + 1)
        subpath = os.path.join(vr_path, f'{pct_remain}')

        # create the new project working directory
        os.mkdir(subpath)
        scaffold_workdir(subpath, include_validation=False)

        # overwrite the processed data directory so we don't need to redo this each time
        shutil.copytree(
            os.path.join(workdir, 'data_processed'),
            os.path.join(subpath, 'data_processed'),
            dirs_exist_ok=True
        )

        # sample the gauge table
        gt = gt.sample(n=n)
        gt.to_csv(os.path.join(subpath, 'gis_inputs', 'gauge_table.csv'), index=False)
        shutil.copyfile(os.path.join(workdir, 'gis_inputs', 'drain_table.csv'),
                        os.path.join(subpath, 'gis_inputs', 'drain_table.csv'))

        # filter the copied processed data to only contain the gauges included in this filtered step
        processed_sim_data = glob.glob(os.path.join(subpath, 'data_processed', 'obs-*.csv'))
        for f in processed_sim_data:
            a = pd.read_csv(f, index_col=0)
            a = a.filter(items=gt['gauge_id'].astype(str))
            a.to_csv(f)

    return


def run_series(workdir: str, drain_shape: str, obs_data_dir: str = None) -> None:
    val_workdirs = [i for i in glob.glob(os.path.join(workdir, 'validation_runs', '*')) if os.path.isdir(i)]
    for val_workdir in val_workdirs:
        print(f'\n\n\t\t\tworking on {val_workdir}\n\n')
        analyze_region(val_workdir, drain_shape, obs_data_dir)
    return
