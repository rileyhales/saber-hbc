import os

import rbc


workdir = '/Users/rchales/data/regional-bias-correction/colombia-magdalena'
drain_shape = os.path.join(workdir, 'gis_inputs', 'magdalena_drainagelines.geojson')

# Only need to do this step 1x ever
rbc.prep.scaffold_working_directory(workdir)

# Create the gauge_table and drain_table.csv
# Scripts not provided, check readme for instructions

# Prepare the observation and simulation data
# Only need to do this step 1x ever
rbc.prep.historical_simulation(os.path.join(workdir, 'data_simulated', 'south_america_era5_qout.nc'), workdir)
rbc.prep.observation_data(workdir)

# Generate the assignments table
rbc.prep.gen_assignments_table(workdir)
