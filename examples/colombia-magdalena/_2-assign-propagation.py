import os

import rbc
import pandas as pd


workdir = '/Users/rchales/data/regional-bias-correction/colombia-magdalena'
drain_shape = os.path.join(workdir, 'gis_inputs', 'magdalena_drainagelines.geojson')

assign_table = pd.read_csv(os.path.join(workdir, 'assign_table.csv'))

# Assign basins which are gauged and propagate those gauges
assign_table = rbc.assign.gauged(assign_table)
assign_table = rbc.assign.propagation(assign_table)

# Cache the assignments table with the updates
rbc.assign.cache_table(workdir, assign_table)

# Generate GIS files so you can go explore your progress graphically
rbc.gis.clip_by_assignment(workdir, assign_table, drain_shape)
