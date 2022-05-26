# assign table and gis_input file required column names
mid_col = 'model_id'
gid_col = 'gauge_id'
asgn_mid_col = 'assigned_model_id'
asgn_gid_col = 'assigned_gauge_id'
down_mid_col = 'downstream_model_id'
reason_col = 'reason'
area_col = 'drain_area'
order_col = 'stream_order'

# name of some files produced by the algorithm
cluster_count_file = 'best-fit-cluster-count.json'
cal_nc_name = 'calibrated_simulated_flow.nc'
sim_ts_pickle = 'sim_time_series.pickle'

# metrics computed on validation sets
metric_list = ['ME', 'MAE', 'RMSE', 'NRMSE (Mean)', 'MAPE', 'NSE', 'KGE (2012)']
metric_nc_name_list = ['ME', 'MAE', 'RMSE', 'NRMSE', 'MAPE', 'NSE', 'KGE2012']
