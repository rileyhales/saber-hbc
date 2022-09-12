# Prepare Spatial Data (scripts not provided)
This step instructs you to collect 3 gis files and use them to generate 2 tables. All 5 files (3 gis files and 2
tables) should go in the `gis_inputs` directory 

1. Clip model drainage lines and catchments shapefile to extents of the region of interest. 
   For speed/efficiency, merge their attribute tables and save as a csv.
   - read drainage line shapefile and with GeoPandas 
   - delete all columns ***except***: NextDownID, COMID, Tot_Drain_, order_
   - rename the columns:
      - NextDownID -> downstream_model_id
      - COMID -> model_id
      - Tot_Drain -> drainage_area
      - order_ -> stream_order
   - compute the x and y coordinates of the centroid of each feature (needs the geometry column)
   - delete geometry column
   - save as `drain_table.csv` in the `gis_inputs` directory

Tip to compute the x and y coordinates using geopandas


Your table should look like this:

| downstream_model_id | model_id        | model_drain_area | stream_order | x   | y   |  
|---------------------|-----------------|------------------|--------------|-----|-----|
| unique_stream_#     | unique_stream_# | area in km^2     | stream_order | ##  | ##  |
| unique_stream_#     | unique_stream_# | area in km^2     | stream_order | ##  | ##  |  
| unique_stream_#     | unique_stream_# | area in km^2     | stream_order | ##  | ##  |  
| ...                 | ...             | ...              | ...          | ... | ... |

1. Prepare a csv of the attribute table of the gauge locations shapefile.
   - You need the columns:
     - model_id
     - gauge_id
     - drainage_area (if known)  

Your table should look like this (column order is irrelevant):

| model_id          | gauge_drain_area | gauge_id         |
|-------------------|------------------|------------------|
| unique_stream_num | area in km^2     | unique_gauge_num |
| unique_stream_num | area in km^2     | unique_gauge_num |
| unique_stream_num | area in km^2     | unique_gauge_num |
| ...               | ...              | ...              |

# Prepare Discharge Data

This step instructs you to gather simulated data and observed data. The raw simulated data (netCDF) and raw observed 
data (csvs) should be included in the `data_inputs` folder. You may keep them in another location and provide the path 
as an argument in the functions that need it. These datasets are used to generate several additional csv files which 
are stored in the `data_processed` directory and are used in later steps. The netCDF file may have any name and the 
directory of observed data csvs should be called `obs_csvs`.

Use the dat

1. Create a single large csv of the historical simulation data with a datetime column and 1 column per stream segment labeled by the stream's ID number.

| datetime   | model_id_1 | model_id_2 | model_id_3 |
|------------|------------|------------|------------|
| 1979-01-01 | 50         | 50         | 50         |
| 1979-01-02 | 60         | 60         | 60         |
| 1979-01-03 | 70         | 70         | 70         |
| ...        | ...        | ...        | ...        |

2. Process the large simulated discharge csv to create a 2nd csv with the flow duration curve on each segment (script provided).

| p_exceed | model_id_1 | model_id_2 | model_id_3 |
|----------|------------|------------|------------|
| 100      | 0          | 0          | 0          |
| 99       | 10         | 10         | 10         |
| 98       | 20         | 20         | 20         |
| ...      | ...        | ...        | ...        |

3. Process the large historical discharge csv to create a 3rd csv with the monthly averages on each segment (script provided).

| month | model_id_1 | model_id_2 | model_id_3 |
|-------|------------|------------|------------|
| 1     | 60         | 60         | 60         |
| 2     | 30         | 30         | 30         |
| 3     | 70         | 70         | 70         |
| ...   | ...        | ...        | ...        |
