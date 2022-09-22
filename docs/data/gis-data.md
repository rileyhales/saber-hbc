# Required GIS Datasets

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

The `drain_table.parquet` should look like this:

| downstream_model_id | model_id        | model_area   | strahler_order | x   | y   |  
|---------------------|-----------------|--------------|----------------|-----|-----|
| unique_stream_#     | unique_stream_# | area in km^2 | stream_order   | ##  | ##  |
| unique_stream_#     | unique_stream_# | area in km^2 | stream_order   | ##  | ##  |  
| unique_stream_#     | unique_stream_# | area in km^2 | stream_order   | ##  | ##  |  
| ...                 | ...             | ...          | ...            | ... | ... |

The `gauge_table.parquet` should look like this:

| model_id          | gauge_id         | gauge_area   |
|-------------------|------------------|--------------|
| unique_stream_num | unique_gauge_num | area in km^2 |
| unique_stream_num | unique_gauge_num | area in km^2 |
| unique_stream_num | unique_gauge_num | area in km^2 |
| ...               | ...              | ...          |


## Things to check

Be sure that both datasets:

- Are in the same projected coordinate system
- Only contain gauges and reaches within the area of interest. Clip/delete anything else.

Other things to consider:

- You may find it helpful to also have the catchments, adjoint catchments, and a watershed boundary polygon for 
  visualization purposes.