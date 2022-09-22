# Required Datasets

SABER requires [GIS Datasets](./gis-data.md) and [Hydrological Datasets](./discharge-data.md).

These datasets ***need to be prepared independently before using `saber-hbc` functions***. You should organize the datasets in a working 
directory that contains 3 subdirectories, as shown below. SABER will expect your inputs to be in the `tables` directory 
with the correct names and will generate many files to populate the `gis` and `clusters` directories. 

Example project directory structure:

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
