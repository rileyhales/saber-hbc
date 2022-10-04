import glob
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold as RKFolds

from .io import gid_col
from .io import metric_nc_name_list
from .io import dir_valid

from .assign import mp_assign_all
from .calibrate import mp_saber

__all__ = ['val_kfolds', 'read_metrics']


def val_kfolds(workdir: str, assign_df: pd.DataFrame, gauge_data: str, hds: str,
               n_splits: int = 5, n_repeats: int = 10, n_processes: int or None = None) -> None:
    """
    Performs a k-fold validation of the calibration results. The validation set is randomly selected from the
    gauges table. The validation set is then removed from the calibration set and the calibration is repeated
    n_repeats times. The validation set is then added back to the calibration set and the calibration is repeated
    n_repeats times. This process is repeated n_folds times.

    Args:
        workdir: the project working directory
        assign_df: a completed assignment dataframe already filled by a successful SABER assignment run
        gauge_data: path to the directory of observed data
        hds: string path to the hindcast streamflow dataset
        n_splits: the number of folds to create in each iteration
        n_repeats: the number of repeats to perform
        n_processes: number of processes to use for multiprocessing, passed to Pool

    Returns:
        pandas.DataFrame
    """
    # make a dataframe to record metrics from each test set
    metrics_df = pd.DataFrame(columns=metric_nc_name_list)

    # subset the assign dataframe to only rows which contain gauges & reset the index
    assign_df = assign_df[assign_df[gid_col].notna()].reset_index(drop=True)

    for i, (train_rows, test_rows) in enumerate(RKFolds(n_splits=n_splits, n_repeats=n_repeats).split(assign_df.index)):
        # make a copy of the dataframe to work with for this iteration
        val_df = assign_df.copy()

        # on the test rows, clear the gid column so that it is not assigned to the gauge it contains
        val_df.loc[test_rows, gid_col] = np.nan

        # make assignments on the training set
        val_df = mp_assign_all(workdir, val_df, n_processes=n_processes)

        # reduce the dataframe to only the test rows (index/row-order gets shuffled by mp assigning)
        val_df = val_df[val_df[gid_col].isna()].reset_index(drop=True)

        # bias correct only at the gauges in the test set
        mp_saber(val_df, hds, gauge_data, os.path.join(workdir, dir_valid), n_processes=n_processes)

        # calculate metrics for the test set
        iter_metrics_df = read_metrics(workdir)

        # append the iteration's metrics to the metrics dataframe
        iter_metrics_df = pd.concat([metrics_df, iter_metrics_df])

        # write the metrics dataframe and the validation dataframe to disk
        iter_metrics_df.to_csv(os.path.join(workdir, dir_valid, f'iter_{i}_metrics.csv'))
        val_df.to_csv(os.path.join(workdir, dir_valid, f'iter_{i}_validation_table.csv'))

        # clear the validation directory for the next iteration
        shutil.rmtree(os.path.join(workdir, dir_valid))
        os.mkdir(os.path.join(workdir, dir_valid))

    # write the metrics dataframe to disk
    metrics_df.to_csv(os.path.join(workdir, dir_valid, 'metrics.csv'))

    return


def read_metrics(workdir: str) -> pd.DataFrame:
    """
    Reads the metrics from the validation directory and returns a dataframe of the metrics

    Args:
        workdir: the project working directory

    Returns:
        pandas.DataFrame
    """
    # todo read the metrics from the corrected csvs, aggregate them, and return a dataframe
    # todo edit the correction functions to output the metrics to a csv
    return
