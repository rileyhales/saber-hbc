import os

import rbc


workdir = '/Users/rchales/data/regional-bias-correction/colombia-magdalena'
drain_shape = os.path.join(workdir, 'gis_inputs', 'clipped_drain_lines', 'magdalena_drainage_lines.shp')

assign_table = rbc.table.read(workdir)

# # Only need to do this step 1x ever
# rbc.prep.scaffold_working_directory(workdir)

# Create the gauge_table and drain_table.csv
# Scripts not provided, check readme for instructions

# # Generate the assignments table
# assign_table = rbc.prep.gen_assignments_table(workdir)
# rbc.table.cache(workdir, assign_table)

# Prepare the observation and simulation data
# Only need to do this step 1x ever
# rbc.prep.historical_simulation(os.path.join(workdir, 'data_simulated', 'south_america_era5_qout.nc'), workdir)
# rbc.prep.observation_data(workdir)

# Generate the clusters using the historical simulation data
# rbc.cluster.generate(workdir)
assign_table = rbc.cluster.summarize(workdir, assign_table)
rbc.table.cache(workdir, assign_table)

# Assign basins which are gauged and propagate those gauges
assign_table = rbc.assign.gauged(assign_table)
assign_table = rbc.assign.propagation(assign_table)
assign_table = rbc.assign.clusters(assign_table)

# Cache the assignments table with the updates
rbc.table.cache(workdir, assign_table)

# Generate GIS files so you can go explore your progress graphically
rbc.gis.clip_by_assignment(workdir, assign_table, drain_shape)
