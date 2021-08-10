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
```

### 1 Prepare Spatial Data (python scripts not provided for these steps)
1. Clip model drainage lines and catchments shapefile to extents of the region of interest
1. Store the shapefiles in the `gis_inputs` directory
1. For speed/efficiency, save the merged attribute table as a csv and delete extra columns from the table
   - read drainage line and  with GeoPandas 
   - delete geometry column
   - delete other columns ***except***: from_node, to_node, COMID, Shape_leng, Tot_Drain_, order_
   - save as `geometry_table.csv` in the root level of the working directory

```
working_directory
   geometry_table.csv
   
   kmeans_models/
   kmeans_images/
   data_simulated/
   data_observed/
   gis_inputs/
      drainageline_shapefile.shp (names vary)
      catchment_shapefile.shp (names vary)
      boundary_shapefile.shp (names vary)
      gauges_shapefile.shp (names vary)
      
      
```

### 2 Prepare Discharge Data -> Create 5 csv files (function available for geoglows data)
1. Create a single large csv of the historical simulation data with a datetime column and 1 columns per stream segment labeled by the stream's ID number.
1. Process the large historical discharge csv to create a 2nd csv with the flow duration curve on each segment (script provided).
1. Normalize the flow duration curve csv by dividing each stream's flow duration curve by that stream's total average flow and save this as a 3rd csv (script provided).
1. Process the large historical discharge csv to create a 4th csv with the monthly averages on each segment (script provided).
1. Normalize the monthly average csv by dividing each stream's monthly averages by the stream's total average flow and save this as a 5th csv (script provided).

```python
import rbc

rbc.prep.historical_simulation(
   '/path/to/historical/simulation/netcdf.nc',
   '/path/to/drain/table.pickle',
   '/path/to/working/directory'
)
```

After this step, you should have a directory of data that looks like this:
```bash
working_directory
  # prepared by user, names vary
  /input_shapefiles
    model_drainagelines.shp
    model_catchments.shp
    gauge_stations.shp
    combined_model_attribute_table.csv

  # prepared by python script
  /data_simulated
    simulation_data.nc
    
    simulated_fdc.csv
    simulated_fdc_normalized.csv
    simulated_monavg.csv
    simulated_monavg_normalized.csv
  
  # prepared by user, names vary
  /data_observed
    observed_fdc.csv
    observed_fdc_normalized.csv
    observed_monavg.csv
    observed_monavg_normalized.csv
    
    # names for these vary
    station1.csv
    station2.csv
    station3.csv
    ...
  
  /kmeans      
```

### 2 K-means clustering (iterative step)
For each of the following, generate and store clusters for many group sizes. I recommend between 4 and 16, perhaps in intervals of 2?
1. Create clusters of the simulated data by flow duration curve.
1. Create clusters of the simulated data by monthly averages.
1. Create clusters of the observed data by flow duration curve.
1. Create clusters of the observed data by monthly averages.

```python
import rbc

rbc.kmeans.generate_clusters(...)
rbc.kmeans.plot_clusters(...)
```

### 3 Assign basins by Location (contains a gauge)
The justification for this is obvious. The observations are the actual streamflow for that basin. 

- If a basin contains a gauge, the simulated basin should use the data from the gauge in that basin.
- The reason listed for this assignment is "gauged"

### 4 Assign basins by Propagation (hydraulically connected to a gauge)
Theory: being up/down stream of the gauge but on the same stream order probably means that the seasonality of the flow is 
probably the same (same FDC), but the monthly average may change depending on how many streams connect with/diverge from the stream. 
This assumption becomes questionable as the stream order gets larger so the magnitude of flows joining the river may be larger, 
be less sensitive to changes in flows up stream, may connect basins with different seasonality, etc.

- Basins that are (1) immediately up or down stream of a gauge (2) on streams of the same order should use that gauged data.
- The reason listed for this assignment is "Propagation-i" where i is the number of stream segments up/down from the gauge the river is.

### 5 Assign basins by spatially refined clusters (spatial interpretation of machine learning results)
This is where you will determine the number of clusters to use from both the observed and simulated data which were generated in a previous step.
The number of clusters needed will depend on the number of gauged basins compared to the number of total basins. It may be influenced by the spatial 
arrangement of the gauges and the distribution of gauges across streams of various orders.

- Spatially compare the locations of basins which were clustered for being similar on their flow duration curve.
- Review assignments spatially. Run tests and view improvements. Adjust clusters and reassign as necessary.

### 6 Assign remaining basins an average
- Identify ungauged basins that were not assigned observed data for corrections.
- Export the resulting csv of assignments.
- Use the csv to guide applying the correction scripts in various applications.
