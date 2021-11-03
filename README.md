# Regional Bias Correction of Large Hydrological Models
This repository contains Python code which can be used to calibrate biased, non-gridded hydrologic models. Most of the 
code in this repository will work on any model's results. The data preprocessing and automated calibration functions 
are programmed to expect data following the GEOGloWS ECMWF Streamflow Service's structure and format.

## Theory
Basins and streams will be used interchangeably to refer to the specific stream subunit.

Large scale hydrologic models are generally not strongly physically based because of the extreme amounts of input data, 
computing power, and computing time that would be required to run it. The model uses the results of other physically based 
models. This is advantageous to us because we rely on the land surface model's behavior of mapping basins with similar 
hydrologically important geophysical physical characteristics to similar discharge forecasts.

This method relies on spatial analysis, machine learning, and statistical refinements. This is a post processing step which 
makes it applicable to many models when the source data, code, or other inputs make the model inaccessible or impractical 
to tweak. We do this in an attempt to avoid requiring the source model data or the ability to run the model. 
Both of those conditions are not always available or practical when dealing with large scale models or datasets.

## Python environment
- python >= 3.7
- numpy
- pandas
- geopandas (https://github.com/geopandas/geopandas)
- scikit-learn
- tslearn (https://github.com/tslearn-team/tslearn)

## Required Data/Inputs 
You need the following data to follow this procedure. The instructions list Shapefile but other vector spatial data 
file formats are acceptable
1. Shapefiles for the Drainage Lines (lines) and Catchments (polygons) in the watershed as used by the hydrological model. 
   The attribute table should contain (at least) the following entries for each feature:
    - (Both) An identifier number
    - (Both) The ID of the next segment/catchment downstream
    - (Drainage lines, preferably both) The stream order of each segment
    - (Catchments, preferably both) Cumulative upstream drainage area
    - (Drainage lines) the x coordinate of the centroid of each feature
    - (Drainage lines) the y coordinate of the centroid of each feature
1. Shapefile of points representing the location of each of the river gauging stations available.
   The attribute table should contain (at least) the following entries for each point
    - A unique ID number or name assigned to the gauge, preferably numeric. Randomly generate unique numeric ID's if they don't exist.
    - The ID of the stream segment in the model which corresponds to that gauge.
1. Shapefile of the boundary (polygon) of the target region for bias calibration.
1. The Shapefiles for the Drainage Lines, Catchments, Gauges/Stations, and boundary are all in the same projection.
1. Historical simulated discharge for each stream segment and for as long (temporally) as is available.
1. Observed discharge data for as many stream reaches as possible within the target region.
1. The units of the simulation and observation data must be in the same units.
1. A working directory folder on the computer where the scripts are going to be run.

## Process
### 1 Create a Working Directory
```python
import rbc
path_to_working_directory = '/my/file/path'
rbc.prep.scaffold_working_directory(path_to_working_directory)
```

Your working directory should exactly like this. 
```
working_directory
    kmeans_models/
    kmeans_images/
    data_simulated/
    data_observed/
    gis_inputs/
    gis_outputs/
```

### 2 Prepare Spatial Data (scripts not provided)
This step instructs you to collect 3 gis files and use them to generate 2 csv tables. All 5 files (3 gis files and 2
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

```python
import geopandas as gpd


def to_x(a):
    return a.x


def to_y(a):
    return a.y


drain_shape = '/path/to/drainagelines/shapefile'
gdf = gpd.read_file(drain_shape)

gdf['x'] = gdf.centroid.apply(to_x)
gdf['y'] = gdf.centroid.apply(to_y)

```

Your table should look like this:

downstream_model_id | model_id          | drainage_area_mod | stream_order  | x   | y   |  
------------------- | ----------------- | ----------------- | ------------- | --- | --- |
unique_stream_#     | unique_stream_#   | area in km^2      | stream_order  | ##  | ##  |
unique_stream_#     | unique_stream_#   | area in km^2      | stream_order  | ##  | ##  |  
unique_stream_#     | unique_stream_#   | area in km^2      | stream_order  | ##  | ##  |  
...                 | ...               | ...               | ...           | ... | ... |

2. Prepare a csv of the attribute table of the gauge locations shapefile.
   - You need the columns:
     - model_id
     - gauge_id
     - drainage_area (if known)  

Your table should look like this (column order is irrelevant):

model_id          | drainage_area_obs | gauge_id  
----------------- | ----------------- | ------------
unique_stream_num | area in km^2      | unique_gauge_num
unique_stream_num | area in km^2      | unique_gauge_num  
unique_stream_num | area in km^2      | unique_gauge_num  
...               | ...               | ...

Your project's working directory now looks like
```
working_directory/
    kmeans_models/
        (empty)
    kmeans_images/
        (empty)
    data_simulated/
        (empty)
    data_observed/
        (empty)
    gis_inputs/
        drain_table.csv
        gauge_table.csv
        drainageline_shapefile.shp (name/gis_format do not need to match)
        catchment_shapefile.shp (name/gis_format do not need to match)
        gauge_shapefile.shp (name/gis_format do not need to match)
    gis_outputs/
        (empty)
```

### 3 Create the Assignments Table
The Assignments Table is the core of the regional bias correction method it is a table which has a column for every 
stream segment in the model and several columns of other information which are filled in during the RBC algorithm. It 
looks like this:

downstream_model_id | model_id          | drainage_area | stream_order | gauge_id  
------------------- | ----------------- | ------------- | ------------ | ----------------
unique_stream_num   | unique_stream_num | area in km^2  | stream_order | unique_gauge_num
unique_stream_num   | unique_stream_num | area in km^2  | stream_order | unique_gauge_num  
unique_stream_num   | unique_stream_num | area in km^2  | stream_order | unique_gauge_num  
...                 | ...               | ...           | ...          | ...

```python
import rbc
workdir = '/path/to/project/directory/'
rbc.prep.gen_assignments_table(workdir)
```

Your project's working directory now looks like
```
working_directory/
    assign_table.csv    <-- New
    
    kmeans_models/
        (empty)
    kmeans_images/
        (empty)
    data_simulated/
        (empty)
    data_observed/
        (empty)
    gis_inputs/
        drain_table.csv
        gauge_table.csv
        drainageline_shapefile.shp (name/gis_format do not need to match)
        catchment_shapefile.shp (name/gis_format do not need to match)
        gauge_shapefile.shp (name/gis_format do not need to match)
    gis_outputs/
        (empty)
```

### 4 Prepare Discharge Data -> Create 5 csv files (function available for geoglows data)
This step instructs you to gather simulated data and observed data. The raw simulated data (netCDF) and raw observed 
data (csvs) should be included in the `data_simulated` and `data_observed` folders respectively. They are used to 
generate several additional csv files which will be used in machine learning algorthims later in the process.

1. Create a single large csv of the historical simulation data with a datetime column and 1 column per stream segment labeled by the stream's ID number.

datetime    | model_id_1  | model_id_2  | model_id_3  
----------- | ----------- | ----------- | ----------- 
1979-01-01  | 50          | 50          |  50          
1979-01-02  | 60          | 60          |  60          
1979-01-03  | 70          | 70          |  70          
...         | ...         | ...         | ...          
   
2. Process the large simulated discharge csv to create a 2nd csv with the flow duration curve on each segment (script provided).

p_exceed    | model_id_1  | model_id_2  | model_id_3   
----------- | ----------- | ----------- | ----------- 
100         | 0           | 0           | 0                    
99          | 10          | 10          | 10                   
98          | 20          | 20          | 20                   
...         | ...         | ...         | ...                  

3. Process the large historical discharge csv to create a 3rd csv with the monthly averages on each segment (script provided).

month       | model_id_1  | model_id_2  | model_id_3   
----------- | ----------- | ----------- | ----------- 
1           | 60          | 60          | 60                   
2           | 30          | 30          | 30                   
3           | 70          | 70          | 70                   
...         | ...         | ...         | ...                  

```python
import rbc

rbc.prep.historical_simulation(
   '/path/to/historical/simulation/netcdf.nc',
   '/path/to/working/directory'
)
```

After this step, you should have a directory of data that looks like this:

```
working_directory/
    assign_table.csv

    kmeans_models/
        (empty)
    kmeans_images/
        (empty)
    data_simulated/         <-- New \/
        obs-fdc.csv
        obs-monavg.csv
        historical_simulation.nc (name varies)
    data_observed/
        obs-fdc.csv
        obs-fdc.pickle
        obs-monavg.csv
        obs-monavg.pickle
        csvs/
            12345.csv
            67890.csv
            ...             <-- New /\
   gis_inputs/
        (same as previous steps...)
   gis_outputs/
        (empty)
```

### 5 K-means clustering
For each of the following, generate and store clusters for many group sizes- between 2 and 12 should be sufficient.
1. Create clusters of the *simulated* data by their flow duration curve.
2. Create clusters of the *simulated* data by their monthly averages.
3. Create clusters of the *observed* data by their flow duration curve.
4. Create clusters of the *observed* data by their monthly averages.
5. Track the error/inertia/residuals for each number of clusters identified.

Use this code:

```python
import rbc

workdir = '/path/to/project/directory/'
rbc.cluster.generate(workdir)
```

This function creates trained kmeans models saved as pickle files, plots (from matplotlib) of what each of the clusters 
look like, and csv files which tracked the inertia (residuals) for each number of clusters. Use the elbow method to 
identify the correct number of clusters to use on each of the 4 datasets clustered.

Your working directory should be updated with the following

```
working_directory/
    assign_table.csv
    
    kmeans_models/
        best-fit-cluster-count.csv
        sim-fdc-inertia.csv
        sim-monavg-inertia.csv
        obs-fdc-inertia.csv
        obs-monavg-inertia.csv
           
        sim-fdc-norm-2-clusters-model.pickle
        sim-fdc-norm-3-clusters-model.pickle
        sim-fdc-norm-4-clusters-model.pickle
        ...
    kmeans_images/
        sim-fdc-norm-2-clusters.png
        sim-fdc-norm-3-clusters.png
        sim-fdc-norm-4-clusters.png
        ...
    data_simulated/
        (same as previous steps...)
    data_observed/
        (same as previous steps...)
    gis_inputs/
        (same as previous steps...)
    gis_outputs/
        (empty)
```

### 6 Assign basins by Location (streams which contain a gauge)
This step involves editing the `assign_table.csv` and but does not change the file structure of the project.

This step uses the information prepared in the previous steps to assign observed streamflow information to modeled 
stream segments which will be used for calibration. This step does not produce any new files but it does edit the 
existing assign_table csv file. After running these lines, use the `rbc.table.cache` function to write the changes to 
disc.

The justification for this is obvious. The observations are the actual streamflow for that basin. 

- If a basin contains a gauge, the simulated basin should use the data from the gauge in that basin.
- The reason listed for this assignment is "gauged"

```python
import rbc

# assign_table = pandas DataFrame (see rbc.table module)
workdir = '/path/to/project/directory/'
assign_table = rbc.table.read(workdir)
rbc.assign.gauged(assign_table)
```

### 7 Assign basins by Propagation (hydraulically connected to a gauge)
This step involves editing the `assign_table.csv` and but does not change the file structure of the project.

Theory: being up/down stream of the gauge but on the same stream order probably means that the seasonality of the flow is 
probably the same (same FDC), but the monthly average may change depending on how many streams connect with/diverge from the stream. 
This assumption becomes questionable as the stream order gets larger so the magnitude of flows joining the river may be larger, 
be less sensitive to changes in flows up stream, may connect basins with different seasonality, etc.

- Basins that are (1) immediately up or down stream of a gauge and (2) on streams of the same order should use that gauged data.
- The reason listed for this assignment is "propagation-{direction}-{i}" where direction is either "upstream" or "downstream" and 
  i is the number of stream segments up/down from the gauge the river is.

```python
import rbc

# assign_table = pandas DataFrame (see rbc.table module)
workdir = '/path/to/project/directory/'
assign_table = rbc.table.read(workdir)
rbc.assign.propagation(assign_table)
```

### 8 Assign basins by Clusters (hydrologically similar basins)
This step involves editing the `assign_table.csv` and but does not change the file structure of the project.

Using the results of the optimal clusters
- Spatially compare the locations of basins which were clustered for being similar on their flow duration curve.
- Review assignments spatially. Run tests and view improvements. Adjust clusters and reassign as necessary.

```python
import rbc

# assign_table = pandas DataFrame (see rbc.table module)
workdir = '/path/to/project/directory/'
assign_table = rbc.table.read(workdir)
rbc.assign.clusters_by_dist(assign_table)
```

### 9 Generate GIS files of the assignments
At any time during these steps you can use the functions in the `rbc.gis` module to create GeoJSON files which you can 
use to visualize the results of this process. These GIS files help you investigate which streams are being selected and 
used at each step. Use this to monitor the results.

```python
import rbc

workdir = '/path/to/project/directory/'
assign_table = rbc.table.read(workdir)
drain_shape = '/my/file/path/'
rbc.gis.clip_by_assignment(workdir, assign_table, drain_shape)
rbc.gis.clip_by_cluster(workdir, assign_table, drain_shape)
rbc.gis.clip_by_unassigned(workdir, assign_table, drain_shape)

# or if you have a specific set of ID's to check on
list_of_model_ids = [123, 456, 789]
rbc.gis.clip_by_ids(workdir, list_of_model_ids, drain_shape)
```



## Analyzing Performance
This bias correction method should adjust all streams toward a more realistic but still not perfect value for discharge. 
The following steps help you to analyze how well this method performed in a given area by iteratively running this 
method on the same set of streams but with an increasing number of randomly selected observed data stations being 
excluded each time. The code provided will help you partition your gauge table into randomly selected subsets.

### Steps
1. Perform the bias correction method with all available observed data.
2. Generate 5 subsets of the gauge table (using provided code)
   1. One with ~90% of the gauges (drop a random 10% of the observed data stations)
   2. One with ~80% of the gauges (drop the same gauges as before ***and*** and additional random 10%)
   3. One with ~70% of the gauges (drop the same gauges as before ***and*** and additional random 10%)
   4. One with ~60% of the gauges (drop the same gauges as before ***and*** and additional random 10%)
   5. One with ~50% of the gauges (drop the same gauges as before ***and*** and additional random 10%)
3. Perform the bias correction method 5 additional times using the 5 new tables created in the previous step. You now
   have 6 separate bias correction instances; 1 with all available observed data and 5 with decreasing amounts of 
   observed data included.
4. For each of the 5 corrected models with observed data withheld, use the provided code to generate plots and maps of 
   the performance metrics. This will compare the best approximation of the bias corrected model data for that instance 
   against the observed data which was withheld from the bias correction process.