import os

import numpy as np

import rbc


np.seterr(all="ignore")
workdir = '/Users/rchales/data/regional-bias-correction/colombia-magdalena'
drain_shape = os.path.join(workdir, 'gis_inputs', 'magdalena_dl_attrname_xy.json')
obs_data_dir = os.path.join(workdir, 'data_inputs', 'obs_csvs')

# Prepare the working directory - only need to do this step 1x ever
# rbc.prep.scaffold_working_directory(workdir)
# Scripts not provided. Consult README.md for instructions
# Create the gauge_table.csv and drain_table.csv
# Put the historical simulation netCDF in the right folder
# Put the observed data csv files in the data_inputs/obs_csvs folder

# # Prepare the observation and simulation data - Only need to do this step 1x ever
# rbc.prep.historical_simulation(workdir)
# rbc.prep.observed_data(workdir)
#
# # Generate the assignments table
# assign_table = rbc.table.gen(workdir)
# rbc.table.cache(workdir, assign_table)
#
# # Generate the clusters using the historical simulation data
# rbc.cluster.generate(workdir)
# assign_table = rbc.cluster.summarize(workdir, assign_table)
# rbc.table.cache(workdir, assign_table)
#
# # Assign basins which are gauged and propagate those gauges
# assign_table = rbc.assign.gauged(assign_table)
# assign_table = rbc.assign.propagation(assign_table)
# assign_table = rbc.assign.clusters_by_dist(assign_table)
#
# # Cache the assignments table with the updates
# rbc.table.cache(workdir, assign_table)
#
# # Generate GIS files so you can go explore your progress graphically
# rbc.gis.clip_by_assignment(workdir, assign_table, drain_shape)
# rbc.gis.clip_by_cluster(workdir, assign_table, drain_shape)
# rbc.gis.clip_by_unassigned(workdir, assign_table, drain_shape)
#
# # Compute the corrected simulation data
# rbc.calibrate_region(workdir, assign_table)

# run the validation study
rbc.validate.sample_gauges(workdir)
rbc.validate.run_series(workdir, drain_shape, obs_data_dir)
