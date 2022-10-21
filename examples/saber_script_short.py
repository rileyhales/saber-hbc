import logging

import numpy as np
import pandas as pd

import saber

np.seterr(all="ignore")

log_path = '/Users/rchales/Projects/geoglows_saber/log.log'
logging.basicConfig(
    level=logging.INFO,
    filename=log_path,
    filemode='a',
    datefmt='%Y-%m-%d %X',
    format='%(asctime)s: %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # USER INPUTS - POPULATE THESE PATHS
    workdir = ''
    x_fdc_train = ''
    x_fdc_all = ''
    drain_gis = ''
    gauge_data = ''
    hindcast_zarr = ''
    # END USER INPUTS

    # Generate Clusters and Plots
    logger.info('Create Clusters and Plots')
    saber.cluster.cluster(workdir, x_fdc_train)
    # Before continuing, review the clustering results and select the best n_clusters for the next function
    saber.cluster.predict_labels(workdir, n_clusters=5, x=pd.read_parquet(x_fdc_all))

    # Generate Assignments Table and Make Assignments
    logger.info('Make Assignments of Ungauged Basins to Gauges')
    assign_df = saber.assign.generate(workdir)
    assign_df = saber.assign.mp_assign_all(workdir, assign_df)

    # Recommended Optional - Generate GIS files to visually inspect the assignments
    logger.info('Generating GIS files')
    saber.gis.create_maps(workdir, assign_df, drain_gis)

    # Recommended Optional - Compute performance metrics
    logger.info('Compute Performance Metrics')
    saber.validate.mp_bootstrap(workdir, assign_df, gauge_data, hindcast_zarr, n_processes=6)
    saber.validate.bootstrap_figures(workdir)

    # # Optional - Compute the Corrected Simulation Data
    # logger.info('Compute Corrected Simulation Data')
    # saber.calibrate.mp_saber(assign_df, hindcast_zarr, gauge_data)

    logger.info('SABER Completed')
