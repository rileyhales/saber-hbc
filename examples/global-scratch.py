import logging

import numpy as np
import pandas as pd

import saber

np.seterr(all="ignore")


def workflow(workdir, hist_sim_nc, obs_data_dir, drain_gis, gauge_gis):
    # scaffold the project working directory
    logger.info('Scaffold project working directory')
    # saber.io.scaffold_workdir(workdir)

    # Prepare input data
    logger.info('Prepare input data')
    # saber.prep.gis_tables(workdir, drain_gis=drain_gis, gauge_gis=gauge_gis)
    # saber.prep.hindcast(workdir, hist_sim_nc, drop_order_1=True)

    # Generate clusters
    logger.info('Generate Clusters')
    saber.cluster.generate(workdir, x=x_fdc)
    logger.info('Cluster Post-Processing and Metrics')
    saber.cluster.summarize_fit(workdir)
    saber.cluster.calc_silhouette(workdir, x=x_fdc, n_clusters=range(2, 8))
    logger.info('Create Plots')
    saber.cluster.plot_clusters(workdir, x=x_fdc)
    saber.cluster.plot_silhouettes(workdir)
    saber.cluster.plot_centers(workdir)
    saber.cluster.plot_fit_metrics(workdir)

    # ALL COMPLETED ABOVE

    # Generate assignments table
    # print('Generate Assignment Table')
    # assign_table = saber.assign.gen(workdir)
    # assign_table = saber.assign.merge_clusters(workdir, assign_table)
    # assign_table = saber.assign.assign_gauged(assign_table)
    # assign_table = saber.assign.assign_propagation(assign_table)
    # assign_table = saber.assign.assign_by_distance(assign_table)
    # saber.io.write_table(assign_table, workdir, 'assign_table')

    # Generate GIS files
    # print('Generate GIS files')
    # saber.gis.clip_by_assignment(workdir, assign_table, drain_gis)
    # saber.gis.clip_by_cluster(workdir, assign_table, drain_gis)
    # saber.gis.clip_by_unassigned(workdir, assign_table, drain_gis)

    # Compute the corrected simulation data
    # print('Starting Calibration')
    # saber.calibrate_region(workdir, assign_table)

    # run the validation study
    # print('Performing Validation')
    # saber.validate.sample_gauges(workdir, overwrite=True)
    # saber.validate.run_series(workdir, drain_shape, obs_data_dir)
    # vtab = saber.validate.gen_val_table(workdir)
    # saber.gis.validation_maps(workdir, gauge_shape, vtab)

    logger.info('SABER Completed')
    return


# basedir = '/Volumes/T7/global-saber/saber-directories-50fdcpts'
# hindcast_ncs = '/Volumes/T7/GEOGloWS/GEOGloWS-Hindcast-20220430'
# drainline_pqs = '/Volumes/T7/GEOGloWS/DrainageParquet'
# grdc_dir = os.path.join(basedir, 'GRDC')
# # grdc_gis = os.path.join(basedir, 'grdc_points.shp')
# grdc_gis = None
#
#
# regions = [
#     'africa',
#     'australia',
#     'central_america',
#     'central_asia',
#     'east_asia',
#     'europe',
#     'islands',
#     'japan',
#     'middle_east',
#     'north_america',
#     'south_asia',
#     'south_america',
#     'west_asia',
# ]
#
# for wdir in natsorted(glob.glob(os.path.join(basedir, '*'))):
#     country = os.path.basename(wdir)
#     print('\n')
#     print(country)
#     hist_sim_nc = os.path.join(hindcast_ncs, f'{country}-geoglows', 'Qout_era5_t640_24hr_19790101to20220430.nc')
#     drain_gis = os.path.join(drainline_pqs, f'{country}-drainagelines.parquet')
#     workflow(wdir, hist_sim_nc, grdc_dir, drain_gis, grdc_gis)


# workdir = '/Users/rchales/Desktop/tmp'
workdir = '/Volumes/T7/geoglows_saber'
logging.basicConfig(
    level=logging.INFO,
    filename='geoglows_saber.log',
    filemode='w',
    datefmt='%Y-%m-%d %X',
    format='%(asctime)s: %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info('Reading prepared data')
# x_fdc = saber.io.read_table(workdir, 'hindcast_fdc_trans').values
x_fdc = pd.read_parquet('/Volumes/T7/geoglows_saber/Backups/hindcast_fdc_transformed_noord1.parquet').values

# logger.info('Scaffold project working directory')
# saber.io.scaffold_workdir(workdir)

# logger.info('Prepare input data')
# saber.prep.gis_tables(workdir, drain_gis=drain_gis, gauge_gis=gauge_gis)
# saber.prep.hindcast(workdir, hist_sim_nc, drop_order_1=True)

# Generate clusters
logger.info('Generate Clusters')
saber.cluster.generate(workdir, x=x_fdc)

logger.info('Cluster Post-Processing and Metrics')
saber.cluster.summarize_fit(workdir)
saber.cluster.calc_silhouette(workdir, x=x_fdc, n_clusters=range(2, 10))

logger.info('Create Plots')
saber.cluster.plot_clusters(workdir, x=x_fdc)
saber.cluster.plot_silhouettes(workdir)
saber.cluster.plot_centers(workdir)
saber.cluster.plot_fit_metrics(workdir)

logger.info('SABER Completed')
