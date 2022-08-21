import os
import glob
import pandas as pd

# assign table and gis_input file required column names
# mid_col = 'model_id'
mid_col = 'COMID'
gid_col = 'gauge_id'
asgn_mid_col = 'assigned_model_id'
asgn_gid_col = 'assigned_gauge_id'
down_mid_col = 'downstream_model_id'
reason_col = 'reason'
area_col = 'drain_area'
order_col = 'stream_order'

# scaffolded folders
tables = 'tables'
data_in = 'data_inputs'
gis_out = 'gis_outputs'
kmeans_out = 'kmeans_outputs'

# name of some files produced by the algorithm
cluster_count_file = 'best-fit-cluster-count.json'
cal_nc_name = 'calibrated_simulated_flow.nc'
hindcast_table = 'hindcast_series_table.parquet'
hindcast_fdc_table = 'hindcast_fdc_table.parquet'
drain_table = 'drain_table.parquet'
gauge_table = 'gauge_table.parquet'

# metrics computed on validation sets
metric_list = ['ME', 'MAE', 'RMSE', 'MAPE', 'NSE', 'KGE (2012)']
metric_nc_name_list = ['ME', 'MAE', 'RMSE', 'MAPE', 'NSE', 'KGE2012']


def read_drain_table(workdir: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(workdir, tables, drain_table))


def read_gauge_table(workdir: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(workdir, tables, gauge_table))


def get_table_path(workdir: str, table_name: str) -> str:
    if table_name == 'hindcast_series':
        return os.path.join(workdir, tables, hindcast_table)
    elif table_name == 'hindcast_fdc':
        return os.path.join(workdir, tables, hindcast_fdc_table)
    elif table_name == 'drain_table':
        return os.path.join(workdir, tables, drain_table)
    elif table_name == 'gauge_table':
        return os.path.join(workdir, tables, gauge_table)


def guess_hindcast_path(workdir: str) -> str:
    return glob.glob(os.path.join(workdir, data_in, '*.nc*'))[0]
