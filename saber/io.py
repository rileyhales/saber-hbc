import os

import pandas as pd

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
    else:
        raise ValueError(f'Unknown table name: {table_name}')


def read_table(workdir: str, table_name: str) -> pd.DataFrame:
    table_path = get_table_path(workdir, table_name)
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
    table_path = get_table_path(workdir, table_name)
    table_format = os.path.splitext(table_path)[-1]
    if table_format == '.parquet':
        return table.to_parquet(table_path)
    elif table_format == '.feather':
        return table.to_feather(table_path)
    elif table_format == '.csv':
        return table.to_csv(table_path)
    else:
        raise ValueError(f'Unknown table format: {table_format}')
