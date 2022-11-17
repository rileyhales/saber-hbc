The Assignments Table is the core of the `saber` python package. It is a table which has a row for every stream segment 
in the model and several columns which are populated by the GIS datasets and by the results of the SABER process.

The Assignments Table is a `pandas` dataframe which is saved to disk as a `parquet` file.

| downstream_model_id | model_id          | drainage_area | stream_order | gauge_id         |
|---------------------|-------------------|---------------|--------------|------------------|
| unique_stream_num   | unique_stream_num | area in km^2  | stream_order | unique_gauge_num |
| unique_stream_num   | unique_stream_num | area in km^2  | stream_order | unique_gauge_num |
| unique_stream_num   | unique_stream_num | area in km^2  | stream_order | unique_gauge_num |
| ...                 | ...               | ...           | ...          | ...              |


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
import saber as saber

# assign_table = pandas DataFrame (see saber.table module)
workdir = '/path/to/project/directory/'
assign_table = saber.table.read(workdir)
saber.assign.gauged(assign_table)
```

### 7 Assign basins by Propagation (hydraulically connected to a gauge)
This step involves editing the `assign_table.csv` and does not change the file structure of the project.

Theory: being up/down stream of the gauge but on the same stream order probably means that the seasonality of the flow is 
probably the same (same FDC), but the monthly average may change depending on how many streams connect with/diverge from the stream. 
This assumption becomes questionable as the stream order gets larger so the magnitude of flows joining the river may be larger, 
be less sensitive to changes in flows up stream, may connect basins with different seasonality, etc.

- Basins that are (1) immediately up or down stream of a gauge and (2) on streams of the same order should use that gauged data.
- The reason listed for this assignment is "propagation-{direction}-{i}" where direction is either "upstream" or "downstream" and 
  i is the number of stream segments up/down from the gauge the river is.

```python
import saber as saber

# assign_table = pandas DataFrame (see saber.table module)
workdir = '/path/to/project/directory/'
assign_table = saber.table.read(workdir)
saber.assign.propagation(assign_table)
```

### 8 Assign basins by Clusters (hydrologically similar basins)
This step involves editing the `assign_table.csv` and but does not change the file structure of the project.

Using the results of the optimal clusters
- Spatially compare the locations of basins which were clustered for being similar on their flow duration curve.
- Review assignments spatially. Run tests and view improvements. Adjust clusters and reassign as necessary.

```python
import saber as saber

# assign_table = pandas DataFrame (see saber.table module)
workdir = '/path/to/project/directory/'
assign_table = saber.table.read(workdir)
saber.assign.clusters_by_dist(assign_table)
```
