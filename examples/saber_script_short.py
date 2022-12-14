import logging

import saber

# USER INPUTS - POPULATE THESE PATHS
config_file = './config.yml'  # Path to the configuration file
log_path = '/Users/rchales/Data/SABER/log.log'  # leave blank to write to console
# END USER INPUTS

logging.basicConfig(
    level=logging.INFO,
    filename=log_path,
    filemode='a',
    datefmt='%Y-%m-%d %X',
    format='%(asctime)s: [%(name)s:%(lineno)d] %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # Read the Config File
    saber.io.read_config(config_file)
    saber.io.init_workdir(overwrite=False)

    # Generate Clusters and Plots
    logger.info('Create Clusters and Plots')
    saber.cluster.cluster()
    # Before continuing, review the clustering results and select the best n_clusters for the next function
    saber.cluster.predict_labels(n_clusters=5)

    # Generate Assignments Table and Propagate from Gauges, Dams/Reservoirs
    logger.info('Generating Assignments Table')
    assign_df = saber.table.init()
    assign_df = saber.table.mp_prop_gauges(assign_df)
    assign_df = saber.table.mp_prop_regulated(assign_df)
    saber.io.write_table(assign_df, 'assign_table')  # cache results

    # Optional - Compute performance metrics at gauged locations
    logger.info('Perform Bootstrap Validation')
    bs_assign_df = saber.bstrap.mp_table(assign_df)
    bs_metrics_df = saber.bstrap.mp_metrics(bs_assign_df)
    saber.bstrap.postprocess_metrics(bs_metrics_df)

    # Optional - Make all assignments
    # logger.info('Make Assignments')
    # assign_df = saber.assign.mp_assign(assign_df)
    # logger.info('Generating GIS files')
    # saber.gis.create_maps(assign_df)

    logger.info('SABER Completed')
