import os

import numpy as np

import saber


np.seterr(all="ignore")

# workdir = ''
# drain_shape = os.path.join(workdir, 'gis_inputs', '')
# gauge_shape = ''
# obs_data_dir = ''
# hist_sim_nc = ''

# COLOMBIA
# workdir = '/Users/rchales/data/saber/colombia-magdalena'
# drain_shape = os.path.join(workdir, 'gis_inputs', 'magdalena_dl_attrname_xy.json')
# gauge_shape = os.path.join(workdir, 'gis_inputs', 'ideam_stations.json')
# obs_data_dir = os.path.join(workdir, 'data_inputs', 'obs_csvs')
# hist_sim_nc = os.path.join(workdir, 'data_inputs', 'south_america_era5_qout.nc')


# Prepare the working directory - only need to do this step 1x ever
# saber.prep.scaffold_working_directory(workdir)
# Scripts not provided. Consult README.md for instructions
# Create the gauge_table.csv and drain_table.csv
# Put the historical simulation netCDF in the right folder
# Put the observed data csv files in the data_inputs/obs_csvs folder

# Prepare the observation and simulation data - Only need to do this step 1x ever
print('Preparing data')
saber.prep.hindcast(workdir)

# Generate the assignments table
print('Generate Assignment Table')
assign_table = saber.table.gen(workdir)
saber.table.cache(workdir, assign_table)

# Generate the clusters using the historical simulation data
print('Generate Clusters')
saber.cluster.generate(workdir)
assign_table = saber.cluster.summarize(workdir, assign_table)
saber.table.cache(workdir, assign_table)

# Assign basins which are gauged and propagate those gauges
print('Making Assignments')
assign_table = saber.assign.gauged(assign_table)
assign_table = saber.assign.propagation(assign_table)
assign_table = saber.assign.clusters_by_dist(assign_table)

# Cache the assignments table with the updates
saber.table.cache(workdir, assign_table)

# Generate GIS files so you can go explore your progress graphically
print('Generate GIS files')
saber.gis.clip_by_assignment(workdir, assign_table, drain_shape)
saber.gis.clip_by_cluster(workdir, assign_table, drain_shape)
saber.gis.clip_by_unassigned(workdir, assign_table, drain_shape)

# Compute the corrected simulation data
print('Starting Calibration')
saber.calibrate_region(workdir, assign_table)

# run the validation study
print('Performing Validation')
saber.validate.sample_gauges(workdir, overwrite=True)
saber.validate.run_series(workdir, drain_shape, obs_data_dir)
vtab = saber.validate.gen_val_table(workdir)
saber.gis.validation_maps(workdir, gauge_shape, vtab)
