import os


# COLOMBIA
workdir = '/Users/rchales/data/regional-bias-correction/colombia-magdalena'
drain_shape = os.path.join(workdir, 'gis_inputs', 'magdalena_dl_attrname_xy.json')
gauge_shape = os.path.join(workdir, 'gis_inputs', 'ideam_stations.json')
obs_data_dir = os.path.join(workdir, 'data_inputs', 'obs_csvs')
hist_sim_nc = os.path.join(workdir, 'data_inputs', 'south_america_era5_qout.nc')

# TEXAS
workdir = '/Users/rchales/data/regional-bias-correction/texas'
drain_shape = os.path.join(workdir, 'shapefiles', 'texas-dl.json')
gauge_shape = os.path.join(workdir, 'shapefiles', 'texas-gauges.shp')
obs_data_dir = os.path.join(workdir, 'data_inputs', 'obs_csvs')
hist_sim_nc = os.path.join(workdir, 'data_inputs', 'Qout_era5_t640_24hr_19790101to20210630.nc.nc')
