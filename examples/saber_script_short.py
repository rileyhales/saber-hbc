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

    # Generate Plots
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

    # Optional - Compute the Corrected Simulation Data
    saber.calibrate.mp_saber(assign_df, hindcast_zarr, gauge_data)

    # Recommended Optional - Compute stochastic performance metrics
    # print('Performing Validation')
    # saber.validate.sample_gauges(workdir, overwrite=True)
    # saber.validate.run_series(workdir, drain_shape, obs_data_dir)
    # vtab = saber.validate.gen_val_table(workdir)
    # saber.gis.validation_maps(workdir, gauge_shape, vtab)

    logger.info('SABER Completed')
