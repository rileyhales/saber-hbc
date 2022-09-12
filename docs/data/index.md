# Required Datasets

## GIS Datasets

1. Drainage lines (usually delineated center lines) with at least the following attributes (columns) 
   for each feature:
    - `model_id`: A unique identifier/ID, any alphanumeric utf-8 string will suffice 
    - `downstream_model_id`: The ID of the next downstream reach 
    - `strahler_order`: The strahler stream order of each reach
    - `model_drain_area`: Cumulative upstream drainage area
    - `x`: The x coordinate of the centroid of each feature (precalculated for faster results later)
    - `y`: The y coordinate of the centroid of each feature (precalculated for faster results later)
2. Points representing the location of each of the river gauging station available with at least the 
   following attributes (columns) for each feature:
    - `gauge_id`: A unique identifier/ID, any alphanumeric utf-8 string will suffice.
    - `model_id`: The ID of the stream segment which corresponds to that gauge.

Be sure that both datasets:

- Are in the same projected coordinate system
- Only contain gauges and reaches within the area of interest. Clip/delete anything else.

Other things to consider:

- You may find it helpful to also have the catchments, adjoint catchments, and a watershed boundary polygon for 
  visualization purposes.

## Hydrological Datasets

1. Hindcast/Retrospective/Historical Simulation for every stream segment (reporting point) in the model. This is a time 
   series of discharge (Q) for each stream segment. The data should be in a tabular format that can be read by `pandas`.
    The data should have two columns:
    1. `datetime`: The datetime stamp for the measurements
    2. A column whose name is the unique `model_id` containing the discharge for each time step.
2. Observed discharge data for each gauge
    1. `datetime`: The datetime stamp for the measurements
    2. A column whose name is the unique `gauge_id` containing the discharge for each time step.

Be sure that both datasets:

- Are in the same units (e.g. m3/s)
- Are in the same time zone (e.g. UTC)
- Are in the same time step (e.g. daily average)
- Do not contain any non-numeric values (e.g. ICE, none, etc.)
- Do not contain rows with missing values (e.g. NaN or blank cells)
- Have been cleaned of any incorrect values (e.g. no negative values)
- Do not contain any duplicate rows

## Working Directory

SABER is designed to read and write many files in a working directory.
                                     
    tables/
        # This directory contains all the input datasets
        drain_table.parquet
        gauge_table.parquet
        model_id_list.parquet
        hindcast_series.parquet
        hindcast_fdc.parquet
        hindcast_fdc_transformed.parquet
    clusters/
        # this directory contains outputs from the SABER commands
        ... 
    gis/
        # this directory contains outputs from the SABER commands
        ...

`drain_table.parquet` is a table of the attribute table from the drainage lines GIS dataset. It can be generated with 
`saber.prep.gis_tables()`.

`gauge_table.parquet` is a table of the attribute table from the drainage lines GIS dataset. It can be generated with 
`saber.prep.gis_tables()`.
