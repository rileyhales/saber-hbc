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
    gauge_gis = ''
    gauge_data = ''
    hindcast_zarr = ''
    n_processes = 1
    # END USER INPUTS

    workdir = '/Users/rchales/Projects/geoglows_saber'
    x_fdc_train = '/Users/rchales/Projects/geoglows_saber/master_copies/hindcast_fdc_transformed_noord1.parquet'
    x_fdc_all = '/Users/rchales/Projects/geoglows_saber/master_copies/hindcast_fdc_transformed_all.parquet'
    drain_gis = '/Users/rchales/Projects/geoglows_saber/geoglows_drain_lines_saber_attributes.gpkg'
    gauge_gis = '/Users/rchales/Data/gauge_discharge_gis/all_gauges.gpkg'
    gauge_data = '/Users/rchales/Data/SABERMASTERS/observed_discharge/observed_discharge'
    hindcast_zarr = '/Users/rchales/Data/geoglows_hindcast/20220430_zarr/*.zarr'
    n_processes = 20

    # # Generate Clusters and Plots
    # logger.info('Create Clusters and Plots')
    # saber.cluster.cluster(workdir, x_fdc_train)
    # # Before continuing, review the clustering results and select the best n_clusters for the next function
    # saber.cluster.predict_labels(workdir, n_clusters=5, x=pd.read_parquet(x_fdc_all))
    #
    # Generate Assignments Table and Propagate from Gauges and Dams/Reservoirs
    # logger.info('Make Assignments of Ungauged Basins to Gauges')
    # assign_df = saber.table.init(workdir)
    # assign_df = saber.table.mp_prop_gauges(assign_df, n_processes=n_processes)
    # assign_df = saber.table.mp_prop_regulated(assign_df, n_processes=n_processes)
    # saber.io.write_table(assign_df, workdir, 'assign_table')
    assign_df = saber.io.read_table(workdir, 'assign_table')

    # # Optional - Make all assignments
    # assign_df = saber.assign.mp_assign(assign_df, n_processes=n_processes)

    # # Optional - Generate GIS files to visually inspect the assignments (must follow mp_assign)
    # logger.info('Generating GIS files')
    # saber.gis.create_maps(workdir, assign_df, drain_gis)

    # Optional - Compute performance metrics at gauged locations
    logger.info('Compute Performance Metrics')
    bs_assign_df = saber.bstrap.mp_table(workdir, assign_df, n_processes=n_processes)
    bs_metrics_df = saber.bstrap.mp_metrics(workdir, bs_assign_df, gauge_data, hindcast_zarr, n_processes=n_processes)
    saber.bstrap.merge_metrics_and_gis(workdir, gauge_gis, bs_metrics_df)

    logger.info('SABER Completed')
