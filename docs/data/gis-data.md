# Required GIS Datasets

You need the following GIS datasets.

1. **Drainage lines** - Polyline layer of the stream centerline locations
2. **(Optional) Catchment boundaries** - Polygon layer of the subbasin boundaries for each stream centerline
3. **Gauge locations** - Point layer of the stream gauge locations

## Drainage Lines (and optional catchment boundaries)
These are the stream center lines used by the hydrologic model. They are usually delineated center lines. You should 
provide these in geopackage format. ESRI Shapefile or other formats readable by geopandas are also acceptable.

Each stream should be a separate feature with at least the following attributes (columns).

 - `model_id`: A unique identifier/ID, any alphanumeric utf-8 string will suffice 
 - `downstream_model_id`: The ID of the next downstream reach 
 - `strahler_order`: The strahler stream order of each reach
 - `model_drain_area`: Cumulative upstream drainage area
 - `x`: The x coordinate of the centroid of each feature (precalculated for faster results later)
 - `y`: The y coordinate of the centroid of each feature (precalculated for faster results later)

It should look like this:

| downstream_model_id | model_id        | model_area   | strahler_order | x   | y   |  
|---------------------|-----------------|--------------|----------------|-----|-----|
| unique_stream_#     | unique_stream_# | area in km^2 | stream_order   | ##  | ##  |
| unique_stream_#     | unique_stream_# | area in km^2 | stream_order   | ##  | ##  |  
| unique_stream_#     | unique_stream_# | area in km^2 | stream_order   | ##  | ##  |  
| ...                 | ...             | ...          | ...            | ... | ... |

## Catchment Boundaries
If you have catchment boundaries, you can use these for alternate visualizations. The catchment boundaries should have the
same properties as the drainage lines and be in the same file formats as the drainage lines.

## Gauge Locations
The locations of each gauge with observered discharge available for bias correction. You need to have already assigned 
each gauge to a corresponding stream segment which represents that point in the hydrologic model.

 - `gauge_id`: A unique identifier/ID, any alphanumeric utf-8 string will suffice.
 - `model_id`: The ID of the stream segment which corresponds to that gauge.
 - `gauge_area`: Optional - the drainage area reported by the gauge

It should look like this:

| model_id          | gauge_id         | gauge_area   |
|-------------------|------------------|--------------|
| unique_stream_num | unique_gauge_num | area in km^2 |
| unique_stream_num | unique_gauge_num | area in km^2 |
| unique_stream_num | unique_gauge_num | area in km^2 |
| ...               | ...              | ...          |


## Things to check

Be sure that all datasets:

- Are in the same projected coordinate system (e.g. EPSG:3857 for web mercator at the global scale)
- Only contain gauges and reaches within the area of interest. Clip/delete anything else.

Other things to consider:

- You may find it helpful to also have the catchments, adjoint catchments, and a watershed boundary polygon for 
  visualization purposes.