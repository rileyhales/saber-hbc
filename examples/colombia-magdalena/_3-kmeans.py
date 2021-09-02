import os

import tslearn.clustering

import rbc

workdir = '/Users/rchales/data/regional-bias-correction/colombia-magdalena'

modelfile = os.path.join(workdir, 'kmeans_models', 'observed_fdc_normalized.csv-6-clusters-model.pickle')

# a = tslearn.clustering.TimeSeriesKMeans.from_pickle(modelfile)
# print()
rbc.kmeans.generate_clusters(workdir)
