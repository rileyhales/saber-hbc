import logging

import saber

# USER INPUTS - POPULATE THESE PATHS
config_file = '/Users/rchales/Projects/saber-hbc/examples/config.yml'  # Path to the configuration file
log_path = ''  # leave blank to write to console
# END USER INPUTS

logging.basicConfig(
    level=logging.INFO,
    filename=log_path,
    filemode='a',
    datefmt='%Y-%m-%d %X',
    format='%(asctime)s: %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    saber.io.read_config(config_file)

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
    saber.io.write_table(assign_df, 'assign_table')

    # Optional - Compute performance metrics at gauged locations
    logger.info('Generate Bootstrap ')
    bs_assign_df = saber.bstrap.mp_table(assign_df)
    bs_metrics_df = saber.bstrap.mp_metrics(bs_assign_df)
    saber.bstrap.merge_metrics_and_gis(bs_metrics_df)

    # Optional - Make all assignments
    logger.info('Make Assignments')
    # assign_df = saber.assign.mp_assign(assign_df, n_processes=n_processes)

    # Optional - Generate GIS files to visually inspect the assignments (must run assignments command first)
    logger.info('Generating GIS files')
    # saber.gis.create_maps(workdir, assign_df, drain_gis)

    logger.info('SABER Completed')
