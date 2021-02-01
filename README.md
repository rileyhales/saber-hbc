# streams
**Large Scale Hydrologic Model Post Process Calibration**

# Python environment
- python >= 3.7
- numpy
- pandas
- geopandas (https://github.com/geopandas/geopandas)
- scikit-learn
- tslearn (https://github.com/tslearn-team/tslearn)

# Required Data/Inputs
1. Shapefiles for the Drainage Lines and Catchments in the watershed. The attribute table should contain:
    - (Both) An identifier number
    - (Both) The ID of the next segment/catchment downstream
    - (Drainage lines) The stream order of each segment
    - (Catchments) Cumulative upstream drainage area
2. Historical hydrologically simulatied discharge for each stream segment for as long as is available. (same units as the observations)
3. Observed discharge data for as many stream reaches as are available. (same units as the simulations)

# Theory
Basins and streams will be used interchangeable to refer to the specific stream subunit.

Large scale models are generally not physically based models because of the extreme amounts of data and computing time that 
would be required to run the model. This is advantageous to us because we rely on the land surface model's behavior of mapping 
basins with similar, hydrologically important, geophysical physical characteristics to similar discharge forecasts.

This method is intended to fit a wide variety of hydrologic models. This method relies on spatial analysis, machine learning, 
and various statistically driven methods in an attempt to avoid requiring the source model data or the ability to run the model. 
Both of those conditions are not always available or practical when dealing with large scale models or datasets.

Each of the process steps described uses csvs but pandas can easily read/write pickles instead which should produce faster read/write times.

# Process
## 1 Data preparation -> Create 6 csv files
1. For speed/efficiency, create a csv of the attribute table of both of shapefiles. Delete extra attribute columns from the csv.
1. Create a single large csv of the historical simulation data with a datetime column and 1 columns per stream segment.
1. Process the large historical discharge csv to create a 2nd csv with the flow duration curve on each segment.
1. Normalize the flow duration curve csv by dividing each stream's flow duration curve by that stream's total average flow and save this as a 3rd csv
1. Process the large historical discharge csv to create a 4th csv with the monthly averages on each segment.
1. Normalize the monthly average csv by dividing each stream's monthly averages by the stream's total average flow and save this as a 5th csv

## 2 K-means clustering (machine learning)
For each of the following, generate and store clusters for many group sizes. I recommend between 4 and 16, perhaps in intervals of 2.
1. Create clusters of the simulated data by flow duration curve.
1. Create clusters of the simulated data by monthly averages.
1. Create clusters of the observed data by flow duration curve.
1. Create clusters of the observed data by monthly averages.

## 3 Assign basins by Location (contains a gauge)
The justification for this is obvious. The observations are the actual streamflow for that basin. 

1. If a basin contains a gauge, the simulated basin should use the data from the gauge in that basin.

## 4 Assign basins by Propagation (hydraulically connected to a gauge)
Theory: being up/down stream of the gauge but on the same stream order probably means that the seasonality of the flow is 
probably the same (same FDC) but the monthly average may change depending on how many streams connect with/diverge from the stream. 
This assumption becomes questionable as the stream order gets larger so the magnitude of flows joining the river may be more 
dramatic and may connect basins with different seasonality. 

1. Basins that are (1) immediately up or down stream of a gauge (2) on streams of the same order should use that gauged data.

## 5 Assign basins by spatially refined clusters (spatial interpretation of machine learning results)
This is where you will determine the number of clusters to use from both the observered and simulated data which were generated in a previous step.
The number of clusters needed will depend on the number of gauged basins compared to the number of total basins and may be influenced by the spatial 
arrangement of the gauges and the distribution of gauges across streams of various order.

1. Spatially compare the locations of basins which were clustered for being similar on their flow duration curve.
1. Review assignments spatially. Run tests and view improvements. Adjust clusters and reassign as necessary.

## 6 Assign remaining basins an average
6. Identify ungauged basins that were not assigned observed data for corrections.
8. Export the resulting csv of assignments.
9. Use the csv to guide applying the correction scripts in various applications.