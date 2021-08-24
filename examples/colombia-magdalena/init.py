import os

import rbc
import pandas as pd
import geopandas as gpd


workdir = '/Users/rchales/data/regional-bias-correction/new-colombia'

# rbc.prep.scaffold_working_directory(working_directory)
rbc.prep.gen_assignments_table(workdir)
# rbc.prep.historical_simulation(os.path.join(workdir, 'data_simulated', 'south_america_era5_qout.nc'), workdir)
a = pd.read_csv(os.path.join(workdir, 'assign_table.csv'))
rbc.assign.gauged(a)

