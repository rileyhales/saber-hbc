mid_col = 'model_id'
gid_col = 'gauge_id'
asgn_mid_col = 'assigned_model_id'
asgn_gid_col = 'assigned_gauge_id'
down_mid_col = 'downstream_model_id'

reason_col = 'reason'
area_col = 'drain_area'
order_col = 'stream_order'

cluster_count_file = 'best-fit-cluster-count.json'

# name of nc with all the simulated and calibrated flows plus metrics
cal_nc_name = 'calibrated_simulated_flow.nc'

metric_list = ["ME", "RMSE", "NRMSE (Mean)", "MAPE", "NSE", "KGE (2009)", "KGE (2012)", "R (Spearman)", "R (Pearson)"]
metric_nc_name_list = ['ME', 'MAE', 'RMSE', 'NRMSE', 'MAPE', 'NSE', 'KGE2009', 'KGE2012', 'R_SP', 'R_PR']
