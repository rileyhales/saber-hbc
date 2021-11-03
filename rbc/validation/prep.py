import os
import shutil

import pandas as pd

from rbc.prep import scaffold_workdir


def sample_gauges(workdir: str) -> None:
    vspath = os.path.join(workdir, 'validation_runs')
    if os.path.exists(vspath):
        shutil.rmtree(vspath)
        os.mkdir(vspath)

    gt = pd.read_csv(os.path.join(workdir, 'gis_inputs', 'gauge_table.csv'))
    rows_to_drop = round(gt.shape[0] / 10)

    for i in range(5):
        n = 436 - rows_to_drop * (i + 1)
        pct_remain = 100 - 10 * (i + 1)
        subpath = os.path.join(vspath, f'{pct_remain}')

        if os.path.exists(subpath):
            shutil.rmtree(subpath)
            os.mkdir(subpath)
        scaffold_workdir(subpath, include_validation=False)

        gt = gt.sample(n=n)
        gt.to_csv(os.path.join(subpath, 'gis_inputs', f'gauge_table_{pct_remain}.csv'), index=False)

    return


def run_validation_series(workdir: str) -> None:
    vspath = os.path.join(workdir, 'validation_runs')
    dirs = [i for i in os.listdir(vspath) if os.path.isdir(i)]
    print(dirs)
    return

workdirectory = '/Users/rchales/data/regional-bias-correction/colombia-magdalena'
sample_gauges(workdirectory)
run_validation_series(workdirectory)
