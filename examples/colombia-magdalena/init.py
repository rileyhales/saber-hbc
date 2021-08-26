import os

import rbc
import pandas as pd


workdir = '/Users/rchales/data/regional-bias-correction/colombia-magdalena'
drain_shapefile = ''

# Only need to do this step 1x ever
# rbc.prep.scaffold_working_directory(workdir)

# Create the gauge_table and drain_table.csv
# Scripts not provided, check readme for instructions

# Prepare the observation and simulation data
# rbc.prep.historical_simulation(os.path.join(workdir, 'data_simulated', 'south_america_era5_qout.nc'), workdir)
# rbc.prep.observation_data(workdir)

# Generate the assignments table
rbc.prep.gen_assignments_table(workdir)
atable = pd.read_csv(os.path.join(workdir, 'assign_table.csv'))

# Assign basins which are gauged and propagate those gauges
atable = rbc.assign.gauged(atable)
atable = rbc.assign.propagation(atable)

# Cache the assignments table with the updates
rbc.assign.cache_table(atable, workdir)

# Generate GIS files so you can go explore your progress graphically
rbc.gis.clip_by_assignment(atable, drain_shapefile, workdir)
