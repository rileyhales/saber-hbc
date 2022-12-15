# Required Datasets

SABER requires GIS and discharge data. These data need to be collected and processed into a standard format the `saber-hbc`
functions know how to process. The hardest part of SABER is preparing these datasets. 

You provide these datasets to `saber-hbc` through a `config.yml`. An example of this file is found in the examples
directory of the source repository and in these docs. These datasets need to be prepared independently of SABER and no
scripts are provided because there are too many possible file formats to account for. An explanation of what each input
dataset should look like is provided to assist in preparing your data.

```yaml
# file paths to input data
workdir: ''

cluster_data: ''

drain_table: ''
gauge_table: ''
regulate_table: ''

drain_gis: ''
gauge_gis: ''

gauge_data: ''
hindcast_zarr: ''

# options for processing data
n_processes: 1
```

## Required Datasets
### `workdir`

The `workdir` should be a path to a directory on your computer where the results of the saber process should be cached.
the `saber-hbc` package will create the necessary subfolders and populate them with files as functions are executed to
produce them. You will need read/write access to this directory and <= 1 GB of free space.

### `cluster_data`

`cluster_data` is a table of training data to cluster the watersheds/subbasins in parquet or csv format. The recommended
features to include are z-score transformed (standard scalar) flow duration curve values for the exceedance
probabilities
from 100 to 0 in increments of 2.5 for a total of 41 features. Many other physical features can be included but were not
as thoroughly investigated during the research of the SABER method. A mockup of the table structure and required
properties
is given below. You can find an example of this table in the zipped sample data.

- It should be a table of data in usual machine learning shape of [n_samples, n_features], or 1 row per feature (
  subbasin) and 1 column
  per feature.
- Each subbasin should be z-scaled individually, not each column of the combined dataset.
- The index is the `model_id` of each subbasin.
- The columns are the features of each subbasin.
- the data type should be float
- the index should be unique

| model_id | Q100 | Q97.5 | Q95 | ... | Q5  | Q2.5 | Q0  |
|----------|------|-------|-----|-----|-----|------|-----|
| 1        | 60   | 50    | 40  | ... | 10  | 5    | 0   |
| 2        | 60   | 50    | 40  | ... | 10  | 5    | 0   |
| 3        | 60   | 50    | 40  | ... | 10  | 5    | 0   |
| 4        | 60   | 50    | 40  | ... | 10  | 5    | 0   |

### `drain_table`
`drain_table` is a table of properties from the stream/catchment network used in the hydrologic model in parquet or csv 
format. The table should have exactly the following features

- `model_id`: A unique identifier/ID, any alphanumeric utf-8 string will suffice 
- `downstream_model_id`: The ID of the next downstream reach, used to trace the network programmatically
- `strahler_order`: The strahler stream order of each reach
- `x`: The x coordinate of the centroid of each feature (precalculated for faster results later)
- `y`: The y coordinate of the centroid of each feature (precalculated for faster results later)

| downstream_model_id | model_id        | model_area   | strahler_order | x   | y   |  
|---------------------|-----------------|--------------|----------------|-----|-----|
| unique_stream_#     | unique_stream_# | area in km^2 | stream_order   | ##  | ##  |
| ...                 | ...             | ...          | ...            | ... | ... |

### `gauge_table`
`gauge_table` is a table of properties from the available river gauges in parquet or csv format. Each gauge needs to have 
a unique ID and needs be paired with the unique ID of the model subbasin. The table must have exactly the following features 
and names:

- `gauge_id`: A unique identifier/ID, any alphanumeric utf-8 string will suffice.
- `model_id`: The ID of the stream segment which corresponds to that gauge.
- `latitude`: The latitude of the gauge
- `longitude`: The longitude of the gauge

| model_id          | gauge_id         | latitude | longitude |
|-------------------|------------------|----------|-----------|
| unique_stream_num | unique_gauge_num | 50       | -20       |
| unique_stream_num | unique_gauge_num | 40       | -70       |
| ...               | ...              | ...      | ...       |

### `regulate_table`
`regulate_table` is a table of about the location of dams, reservoirs, or other regulatory structures in parquet or csv 
format. Each dam needs to have a unique ID and must be paired with the unique ID of the model subbasin that contains it. 
The table must have exactly the following features and names:

- `regulate_id`: A unique identifier/ID, any alphanumeric utf-8 string will suffice.
- `model_id`: The ID of the stream segment which corresponds to this regulatory structure.

| model_id          | regulate_id         |
|-------------------|---------------------|
| unique_stream_num | unique_regulate_num |
| unique_stream_num | unique_regulate_num |
| ...               | ...                 |

### `drain_gis`
`drain_gis` is a geopackage or equivalent of the stream network used in the hydrologic model. It should have exactly the 
same properties listed in the `drain_table` above. This information is the same but can be used to make maps of the 
network and the SABER results.

### `gauge_gis`
`gauge_gis` is a geopackage or equivalent of the river gauges used for validation. It should have exactly the same
properties listed in the `gauge_table` above. This information is the same but can be used to make maps of the gauges and
the SABER results.

| model_id          | gauge_id         | latitude | longitude |
|-------------------|------------------|----------|-----------|
| unique_stream_num | unique_gauge_num | 50       | -20       |
| unique_stream_num | unique_gauge_num | 40       | -70       |
| ...               | ...              | ...      | ...       |

### `gauge_data`
`gauge_data` is a directory that contains the observed discharge information. It should contain a subdirectory called 
`observed_data` which contains the csv files of the river gauge data. Each file should be named with the `gauge_id` of
the gauge it corresponds to. Other extra gis datasets and the gauge table may be included here as well.

### `hindcast_zarr`
`hindcast_zarr` is a zarr file (directory). The SABER code expects this to be a series of zarr files which cover the same 
historical time frame but in separate chunks. SABER was developed to use zarr files that were converted from RAPID netCDF 
outputs. Each file should have globally unique IDs and the zarr's are concatenated along the rivid dimension.


## Processing Options

### `n_processes`
`n_processes` is an integer that specifies the number of processes to use for parallel processing. SABER computations are 
operations on dataframes which are easily parallelizable. This is the number of works used in a Python multiprocessing 
`Pool`. This number should probably be <= the number of cores on your machine. 

## FAQ, Tips, Troubleshooting

### GIS Datasets

Be sure that all gis datasets:

- Are in the same projected coordinate system
- Only contain gauges and reaches within the area of interest. Clip/delete anything else.
- You may find it helpful to also have the catchments, adjoint catchments, and a watershed boundary polygon for 
  visualization purposes.

### Discharge Datasets

Be sure that all the discharge datasets (simulated and observed):

- Are in the same units (e.g. m3/s)
- Are in the same time zone (e.g. UTC)
- Are in the same time step (e.g. daily average)
- Do not contain any non-numeric values (e.g. ICE, none, etc.)
- Do not contain any negative values (e.g. -9999, etc.)
- If the negative values are fill values for null, commonly -9999, delete them from the dataset
- Do not contain rows with missing values (e.g. NaN or blank cells)
- Have been cleaned of any other incorrect values (e.g. contain letters)
- Do not contain any duplicate rows
