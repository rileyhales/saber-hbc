# Required Datasets

SABER requires [GIS Datasets](./gis-data.md) and [Hydrological Datasets](./discharge-data.md). 
***These datasets need to be prepared independently before using `saber-hbc` functions***. 

## GIS Datasets
You need to provide the following GIS datasets. Refer to [GIS Datasets](./gis-data.md) for full details. For faster 
computations and file I/O, you should create a flat table in parquet format of the attribute tables for each dataset.

- **Drainage lines** - Polyline layer of the stream centerline locations
- **Catchment boundaries** - Polygon layer of the subbasin boundaries for each stream centerline
- **Gauge locations** - Point layer of the stream gauge locations

## Hydrological Datasets

- **Simulated Discharge** - hindcast discharge for every stream in Zarr format
- **Observed Discharge** - observed discharge for every gauge in CSV format
- **Simulated Discharge Flow Duration Curves** - hindcast discharge flow duration curves for every stream in Parquet format (calculated from the hindcast dataset)
- **Transformed Simulated FDC** - the flow duration curves transformed by the standard scalar and prepared for clustering in Parquet format (calculated from the FDC dataset)

## Project Working Directory 

You should organize the datasets in a working directory that contains 4 subdirectories as shown below. SABER will expect 
your inputs to be in the `tables` directory with the correct names and will generate many files to populate the 
`gis`, `clusters`, and `validation` directories. 

The required GIS datasets do not need to be included in the project working directory but you need to provide the parquet 
format attribute tables for each dataset in the `tables` directory.

    tables/
        # This directory contains all the input datasets
        drain_table.parquet
        gauge_table.parquet
        model_id_list.parquet
        hindcast_series.parquet
        hindcast_fdc.parquet
        hindcast_fdc_transformed.parquet
    clusters/
        # this directory will be filled with outputs from SABER commands
        ... 
    gis/
        # this directory will be filled with outputs from SABER commands
        ...
    validation/
        # this directory will be filled with outputs from SABER commands
        ...
