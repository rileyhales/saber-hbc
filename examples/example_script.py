import os

import rbc


workdir = '/Users/rchales/data/regional-bias-correction/colombia-magdalena'
drain_shape = os.path.join(workdir, 'gis_inputs', 'magdalena_dl_attrname_xy.json')

# Prepare the working directory - only need to do this step 1x ever
# rbc.prep.scaffold_working_directory(workdir)
# Create the gauge_table.csv and drain_table.csv
# Put the historical simulation netCDF in the right folder
# Put the observed data csv files in the data_observed/csvs folder
# Scripts not provided. Consult README.md for instructions

# Generate the assignments table
assign_table = rbc.table.gen(workdir)
rbc.table.cache(workdir, assign_table)

# Prepare the observation and simulation data - Only need to do this step 1x ever
rbc.prep.historical_simulation(workdir)
rbc.prep.observed_data(workdir)

# Generate the clusters using the historical simulation data
rbc.cluster.generate(workdir)
assign_table = rbc.cluster.summarize(workdir, assign_table)
rbc.table.cache(workdir, assign_table)

# Assign basins which are gauged and propagate those gauges
assign_table = rbc.assign.gauged(assign_table)
assign_table = rbc.assign.propagation(assign_table)
assign_table = rbc.assign.clusters_by_dist(assign_table)

# Cache the assignments table with the updates
rbc.table.cache(workdir, assign_table)

# Generate GIS files so you can go explore your progress graphically
rbc.gis.clip_by_assignment(workdir, assign_table, drain_shape)
rbc.gis.clip_by_cluster(workdir, assign_table, drain_shape)
rbc.gis.clip_by_unassigned(workdir, assign_table, drain_shape)

# Compute the corrected simulation data
rbc.create_archive(workdir, assign_table)
