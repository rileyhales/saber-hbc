from .prep import historical_simulation, hist_sim_table, observed_data

from .table import gen, cache
from .cluster import generate, summarize
from .assign import gauged, propagation, clusters_by_dist
from .gis import clip_by_assignment, clip_by_cluster, clip_by_unassigned
from ._calibrate import calibrate_region


def prep_region(workdir: str) -> None:
    historical_simulation(workdir)
    hist_sim_table(workdir)
    observed_data(workdir)
    return


def analyze_region(workdir: str, drain_shape: str, obs_data_dir: str = None) -> None:
    # Generate the assignments table
    print("gen assign_table")
    assign_table = gen(workdir)
    cache(workdir, assign_table)

    # Generate the clusters using the historical simulation data
    print("working on clustering")
    generate(workdir)
    assign_table = summarize(workdir, assign_table)
    cache(workdir, assign_table)

    # Assign basins which are gauged and propagate those gauges
    print("make assignments")
    assign_table = gauged(assign_table)
    assign_table = propagation(assign_table)
    assign_table = clusters_by_dist(assign_table)

    # Cache the assignments table with the updates
    print("cache table results")
    cache(workdir, assign_table)

    # Generate GIS files so you can go explore your progress graphically
    print("generate gis files")
    clip_by_assignment(workdir, assign_table, drain_shape)
    clip_by_cluster(workdir, assign_table, drain_shape)
    clip_by_unassigned(workdir, assign_table, drain_shape)

    # Compute the corrected simulation data
    print("calibrating region netcdf")
    calibrate_region(workdir, assign_table, obs_data_dir)
    return
