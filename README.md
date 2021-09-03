# Regional Bias Correction of Large Hydrological Models

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

Your working directory should like this
```
working_directory
    kmeans_models/
    kmeans_images/
    data_simulated/
    data_observed/
    gis_inputs/
    gis_outputs/
```

### 1 Prepare Spatial Data (scripts not provided)
1. Clip model drainage lines and catchments shapefile to extents of the region of interest. 
   For speed/efficiency, merge their attribute tables and save as a csv.
   - read drainage line shapefile and with GeoPandas 
   - delete all columns ***except***: NextDownID, COMID, Tot_Drain_, order_
   - rename the columns:
      - NextDownID -> downstream_model_id
      - COMID -> model_id
      - Tot_Drain -> drainage_area
      - order_ -> stream_order
   - delete geometry column
   - save as `drain_table.csv` in the `gis_inputs` directory

Your table should look like this:

downstream_model_id | model_id          | drainage_area_mod | stream_order  
------------------- | ----------------- | ----------------- | ------------
unique_stream_num   | unique_stream_num | area in km^2      | stream_order
unique_stream_num   | unique_stream_num | area in km^2      | stream_order  
unique_stream_num   | unique_stream_num | area in km^2      | stream_order  
...                 | ...               | ...               | ...

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
        drainageline_shapefile.shp
        catchment_shapefile.shp
    gis_outputs/
        (empty)
```

### 2 Create the Assignments Table
The Assignments Table is the core of the regional bias correction method it is a table which has a column for every 
stream segment in the model and several columns of other information which are filled in during the RBC algorithm. It 
looks like this:

downstream_model_id | model_id          | drainage_area | stream_order | gauge_id  
------------------- | ----------------- | ------------- | ------------ | ------------
unique_stream_num   | unique_stream_num | area in km^2  | stream_order | unique_gauge_numb
unique_stream_num   | unique_stream_num | area in km^2  | stream_order | unique_gauge_numb  
unique_stream_num   | unique_stream_num | area in km^2  | stream_order | unique_gauge_numb  
...                 | ...               | ...           | ...          | ...

```python
import rbc
workdir = '/path/to/project/directory/'
rbc.prep.gen_assignments_table(workdir)
```

Your project's working directory now looks like
```
working_directory/
    assign_table.csv
    
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
        drainageline_shapefile.shp
        catchment_shapefile.shp
    gis_outputs/
        (empty)
```

### 2 Prepare Discharge Data -> Create 5 csv files (function available for geoglows data)
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
    data_simulated/
        obs-fdc.csv
        obs-fdc.pickle
        obs-monavg.csv
        obs-monavg.pickle
        (historical_simulation.nc, optional)
    data_observed/
        obs-fdc.csv
        obs-fdc.pickle
        obs-monavg.csv
        obs-monavg.pickle
        (directory of raw observation data, optional)
   gis_inputs/
        drain_table.csv
        gauge_table.csv
        drainageline_shapefile.shp
        catchment_shapefile.shp
   gis_outputs/
        (empty)
```

### 3 K-means clustering
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
        ...
    data_observed/
        ...
    gis_inputs/
        ...
    gis_outputs/
        (empty)
```

### 4 Assign basins by Location (streams which contain a gauge)
The justification for this is obvious. The observations are the actual streamflow for that basin. 

- If a basin contains a gauge, the simulated basin should use the data from the gauge in that basin.
- The reason listed for this assignment is "gauged"

```python
import rbc
workdir = '/path/to/project/directory/'
rbc.assign.gauged(workdir)
```

### 5 Assign basins by Propagation (hydraulically connected to a gauge)
Theory: being up/down stream of the gauge but on the same stream order probably means that the seasonality of the flow is 
probably the same (same FDC), but the monthly average may change depending on how many streams connect with/diverge from the stream. 
This assumption becomes questionable as the stream order gets larger so the magnitude of flows joining the river may be larger, 
be less sensitive to changes in flows up stream, may connect basins with different seasonality, etc.

- Basins that are (1) immediately up or down stream of a gauge and (2) on streams of the same order should use that gauged data.
- The reason listed for this assignment is "propagation-{direction}-{i}" where direction is either "upstream" or "downstream" and 
  i is the number of stream segments up/down from the gauge the river is.

```python
import os
import rbc
workdir = '/path/to/project/directory/'
assign_table = os.path.join(workdir, 'assign_table.csv')
rbc.assign.propagation(assign_table)
```

### 6 Assign basins by Clusters (hydrologically similar basins)
Using the results of the optimal clusters
- Spatially compare the locations of basins which were clustered for being similar on their flow duration curve.
- Review assignments spatially. Run tests and view improvements. Adjust clusters and reassign as necessary.

### 7 Assign remaining basins an average
- Identify ungauged basins that were not assigned observed data for corrections.
- Export the resulting csv of assignments.
- Use the csv to guide applying the correction scripts in various applications.

### 8 Compute the 