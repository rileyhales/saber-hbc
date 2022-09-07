import os

import numpy as np

import saber

np.seterr(all="ignore")

workdir = ''
hist_sim_nc = os.path.join(workdir, '')
obs_data_dir = os.path.join(workdir, '')
drain_gis = os.path.join(workdir, '')
gauge_gis = os.path.join(workdir, '')

print('Prepare inputs')
saber.prep.gis_tables(workdir, drain_gis=drain_gis, gauge_gis=gauge_gis)
saber.prep.hindcast(workdir)

# Generate the assignments table
print('Generate Assignment Table')
assign_table = saber.table.gen(workdir)
saber.table.cache(workdir, assign_table)

# Generate clusters using the historical simulation data
print('Generate Clusters')
saber.cluster.generate(workdir)
saber.cluster.plot(workdir)

# Assign basins which are gauged and propagate those gauges
print('Making Assignments')
assign_table = saber.table.merge_clusters(workdir, assign_table, n_clusters=5)
assign_table = saber.table.assign_gauged(assign_table)
assign_table = saber.table.assign_propagation(assign_table)
assign_table = saber.table.assign_by_distance(assign_table)

# Cache the assignments table with the updates
saber.table.cache(workdir, assign_table)

# Generate GIS files to explore your progress graphically
print('Generate GIS files')
saber.gis.clip_by_assignment(workdir, assign_table, drain_gis)
saber.gis.clip_by_cluster(workdir, assign_table, drain_gis)
saber.gis.clip_by_unassigned(workdir, assign_table, drain_gis)
