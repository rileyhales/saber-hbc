import logging
import os
import pandas as pd
import saber
# USER INPUTS - POPULATE THESE PATHS
config_file = '/Users/yubinbaaniya/Documents/SABER/saber-hbc/examples/config.yml'  # Path to the configuration file
log_path = '' # leave blank to write to console
#the below two line is added by me so that I don't have to run it again

#assign_df_file = '/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/tables/assign_table_cluster7_2nditeration.parquet'  # Path to the saved assign_df file
#assign_table_bootstrap_file = '/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/tables/assign_table_bootstrap_climate and gauge cluster ra clstr seperately same.csv'
# # bs_metrics_df_file = '/Users/yubinbaaniya/Documents/WORLD BIAS/saber workdir/tables/bootstrap_metrics.csv'
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


   # Load already created data
   #  if os.path.exists(assign_df_file):
   #      logger.info(f'Loading existing cluster centroid from {assign_df_file}')
   #      assign_df = pd.read_parquet(assign_df_file)
   #      # print(assign_df)
   #  else:
   #      raise FileNotFoundError(f"The file {assign_df_file} does not exist.  provide the correct path.")

    # if os.path.exists(assign_table_bootstrap_file):
    #     logger.info(f'Loading existing bs_assign_df from {assign_table_bootstrap_file}')
    #     bs_assign_df = pd.read_csv(assign_table_bootstrap_file)
    # else:
    #     logger.info('Performing Bootstrap Validation')
    #     bs_assign_df = saber.bs.mp_table(assign_df)

#    if os.path.exists(bs_metrics_df_file):
#         logger.info(f'Loading existing assign_df from {bs_metrics_df_file}')
#         bs_metrics_df = pd.read_csv(bs_metrics_df_file)
#     else:
#         raise FileNotFoundError(f"The file {bs_metrics_df_file} does not exist. provide the correct path.")

    # Read the Config File
    saber.io.read_config(config_file)
    saber.io.init_workdir(overwrite=False)

    # Generate Clusters and Plots
    # logger.info('Create Clusters and Plots')
    # saber.cluster.cluster()
    # # Before continuing, review the clustering results and select the best n_clusters for the next function
    # saber.cluster.predict_labels(n_clusters=5)
    #
    # Generate Assignments Table and Propagate from Gauges, Dams/Reservoirs
    logger.info('Generating Assignments Table')
    assign_df = saber.table.init()
    assign_df = saber.table.mp_prop_gauges(assign_df)
    assign_df = saber.table.mp_prop_regulated(assign_df)
    saber.io.write_table(assign_df, 'assign_table')  # NOTE THAT THE MODEL_ID ON ASSIGN TABLE SHOULD BE int. So you might have to add that line somewhere here beofre it gove you error because the code produce that column datatype as object
    print(assign_df)# cache results

  #  Optional - Compute performance metrics at gauged locations
    logger.info('Perform Bootstrap Validation')
    bs_assign_df = saber.bs.mp_table(assign_df) # gives dataframe containing gauge and  rprop,gprop,reason
    bs_metrics_df = saber.bs.mp_metrics(bs_assign_df)
    saber.bs.postprocess_metrics(bs_metrics_df)
    saber.bs.histograms()
    saber.bs.pie_charts()

    # # # Optional - Make all assignments
    logger.info('Make Assignments')
    #assign_df = saber.assign.mp_assign(assign_df)
    # logger.info('Generating GIS files')
    # saber.gis.create_maps(assign_df)
    # logger.info('SABER Completed')
