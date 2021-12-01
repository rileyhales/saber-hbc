import os

import numpy as np

import hbc


np.seterr(all="ignore")

workdir = ''
drain_shape = os.path.join(workdir, 'gis_inputs', '')
gauge_shape = ''
obs_data_dir = ''
hist_sim_nc = ''

# Prepare the working directory - only need to do this step 1x ever
# hbc.prep.scaffold_working_directory(workdir)
# Scripts not provided. Consult README.md for instructions
# Create the gauge_table.csv and drain_table.csv
# Put the historical simulation netCDF in the right folder
# Put the observed data csv files in the data_inputs/obs_csvs folder

# Prepare the observation and simulation data - Only need to do this step 1x ever
print('Preparing data')
hbc.prep.historical_simulation(workdir)

# Generate the assignments table
print('Generate Assignment Table')
assign_table = hbc.table.gen(workdir)
hbc.table.cache(workdir, assign_table)

# Generate the clusters using the historical simulation data
print('Generate Clusters')
hbc.cluster.generate(workdir)
assign_table = hbc.cluster.summarize(workdir, assign_table)
hbc.table.cache(workdir, assign_table)

# Assign basins which are gauged and propagate those gauges
print('Making Assignments')
assign_table = hbc.assign.gauged(assign_table)
assign_table = hbc.assign.propagation(assign_table)
assign_table = hbc.assign.clusters_by_dist(assign_table)

# Cache the assignments table with the updates
hbc.table.cache(workdir, assign_table)

# Generate GIS files so you can go explore your progress graphically
print('Generate GIS files')
hbc.gis.clip_by_assignment(workdir, assign_table, drain_shape)
hbc.gis.clip_by_cluster(workdir, assign_table, drain_shape)
hbc.gis.clip_by_unassigned(workdir, assign_table, drain_shape)

# Compute the corrected simulation data
print('Starting Calibration')
hbc.calibrate_region(workdir, assign_table)

# run the validation study
print('Performing Validation')
hbc.validate.sample_gauges(workdir, overwrite=True)
hbc.validate.run_series(workdir, drain_shape, obs_data_dir)
vtab = hbc.validate.gen_val_table(workdir)
hbc.gis.validation_maps(workdir, gauge_shape, vtab)
