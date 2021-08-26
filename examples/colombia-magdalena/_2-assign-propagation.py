import os

import rbc
import pandas as pd


workdir = '/Users/rchales/data/regional-bias-correction/colombia-magdalena'
drain_shape = os.path.join(workdir, 'gis_inputs', 'magdalena_drainagelines.geojson')

atable = pd.read_csv(os.path.join(workdir, 'assign_table.csv'))

# Assign basins which are gauged and propagate those gauges
atable = rbc.assign.gauged(atable)
atable = rbc.assign.propagation(atable)

# Cache the assignments table with the updates
rbc.assign.cache_table(atable, workdir)

# Generate GIS files so you can go explore your progress graphically
rbc.gis.clip_by_assignment(atable, drain_shape, workdir)
