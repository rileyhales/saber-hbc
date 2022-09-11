import glob
import os
from collections.abc import Iterable
from typing import List

import pandas as pd
from natsort import natsorted

# assign table and gis_input file required column names
mid_col = 'model_id'
gid_col = 'gauge_id'
asgn_mid_col = 'assigned_model_id'
asgn_gid_col = 'assigned_gauge_id'
down_mid_col = 'downstream_model_id'
reason_col = 'reason'
area_col = 'drain_area'
order_col = 'strahler_order'

# name of some files produced by the algorithm
cluster_count_file = 'best-fit-cluster-count.json'
cal_nc_name = 'calibrated_simulated_flow.nc'

# scaffolded folders
dir_tables = 'tables'
dir_gis = 'gis'
dir_clusters = 'clusters'
dir_valid = 'validation'

# name of the required input tables and the outputs
table_hindcast = 'hindcast.parquet'
table_hindcast_fdc = 'hindcast_fdc.parquet'
table_hindcast_fdc_trans = 'hindcast_fdc_transformed.parquet'
table_mids = 'mid_table.parquet'
table_drain = 'drain_table.parquet'
table_gauge = 'gauge_table.parquet'
table_assign = 'assign_table.csv'

# tables generated by the clustering functions
table_cluster_metrics = 'cluster_metrics.csv'
table_cluster_sscores = 'cluster_sscores.csv'
table_cluster_labels = 'cluster_labels.parquet'

# metrics computed on validation sets
metric_list = ['ME', 'MAE', 'RMSE', 'MAPE', 'NSE', 'KGE (2012)']
metric_nc_name_list = ['ME', 'MAE', 'RMSE', 'MAPE', 'NSE', 'KGE2012']


def scaffold_workdir(path: str, include_validation: bool = True) -> None:
    """
    Creates the correct directories for a Saber project within the specified directory

    Args:
        path: the path to a directory where you want to create workdir subdirectories
        include_validation: boolean, indicates whether to create the validation folder

    Returns:
        None
    """
    dir_list = [dir_tables, dir_gis, dir_clusters]
    if not os.path.exists(path):
        os.mkdir(path)
    if include_validation:
        dir_list.append(dir_valid)
    for d in dir_list:
        p = os.path.join(path, d)
        if not os.path.exists(p):
            os.mkdir(p)
    return


def get_table_path(workdir: str, table_name: str) -> str:
    """
    Get the path to a table in the project directory by name

    Args:
        workdir:
        table_name:

    Returns:
        Path (str) to the table

    Raises:
        ValueError: if the table name is not recognized
    """
    if table_name == 'hindcast':
        return os.path.join(workdir, dir_tables, table_hindcast)
    elif table_name == 'hindcast_fdc':
        return os.path.join(workdir, dir_tables, table_hindcast_fdc)
    elif table_name == 'hindcast_fdc_trans':
        return os.path.join(workdir, dir_tables, table_hindcast_fdc_trans)
    elif table_name == 'model_ids':
        return os.path.join(workdir, dir_tables, table_mids)
    elif table_name == 'drain_table':
        return os.path.join(workdir, dir_tables, table_drain)
    elif table_name == 'gauge_table':
        return os.path.join(workdir, dir_tables, table_gauge)
    elif table_name == 'assign_table':
        return os.path.join(workdir, dir_tables, table_assign)

    elif table_name == 'cluster_metrics':
        return os.path.join(workdir, dir_clusters, table_cluster_metrics)
    elif table_name == 'cluster_sscores':
        return os.path.join(workdir, dir_clusters, table_cluster_sscores)
    elif table_name == 'cluster_labels':
        return os.path.join(workdir, dir_clusters, table_cluster_labels)
    elif table_name.startswith('cluster_centers_'):  # cluster_centers_{n_clusters}.parquet - 1 per cluster
        return os.path.join(workdir, dir_clusters, f'{table_name}.parquet')
    elif table_name.startswith('cluster_sscores_'):  # cluster_sscores_{n_clusters}.parquet - 1 per cluster
        return os.path.join(workdir, dir_clusters, f'{table_name}.parquet')

    else:
        raise ValueError(f'Unknown table name: {table_name}')


def read_table(workdir: str, table_name: str) -> pd.DataFrame:
    """
    Read a table from the project directory by name.

    Args:
        workdir: path to the project directory
        table_name: name of the table to read

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError: if the table does not exist in the correct directory with the correct name
        ValueError: if the table format is not recognized
    """
    table_path = get_table_path(workdir, table_name)
    if not os.path.exists(table_path):
        raise FileNotFoundError(f'Table does not exist: {table_path}')

    table_format = os.path.splitext(table_path)[-1]
    if table_format == '.parquet':
        return pd.read_parquet(table_path, engine='fastparquet')
    elif table_format == '.feather':
        return pd.read_feather(table_path)
    elif table_format == '.csv':
        return pd.read_csv(table_path)
    else:
        raise ValueError(f'Unknown table format: {table_format}')


def write_table(table: pd.DataFrame, workdir: str, table_name: str) -> None:
    """
    Write a table to the correct location in the project directory

    Args:
        table: the pandas DataFrame to write
        workdir: the path to the project directory
        table_name: the name of the table to write

    Returns:
        None

    Raises:
        ValueError: if the table format is not recognized
    """
    table_path = get_table_path(workdir, table_name)
    table_format = os.path.splitext(table_path)[-1]
    if table_format == '.parquet':
        return table.to_parquet(table_path)
    elif table_format == '.feather':
        return table.to_feather(table_path)
    elif table_format == '.csv':
        return table.to_csv(table_path, index=False)
    else:
        raise ValueError(f'Unknown table format: {table_format}')


def _find_model_files(workdir: str, n_clusters: int or Iterable = 'all') -> List[str]:
    """
    Find all the kmeans model files in the project directory.

    Args:
        workdir: path to the project directory

    Returns:
        List of paths to the kmeans model files

    Raises:
        TypeError: if n_clusters is not an int, iterable of int, or 'all'
    """
    kmeans_dir = os.path.join(workdir, dir_clusters)
    if n_clusters == 'all':
        return natsorted(glob.glob(os.path.join(kmeans_dir, 'kmeans-*.pickle')))
    elif isinstance(n_clusters, int):
        return glob.glob(os.path.join(kmeans_dir, f'kmeans-{n_clusters}.pickle'))
    elif isinstance(n_clusters, Iterable):
        return natsorted([os.path.join(kmeans_dir, f'kmeans-{i}.pickle') for i in n_clusters])
    else:
        raise TypeError('n_clusters should be of type int or an iterable')
