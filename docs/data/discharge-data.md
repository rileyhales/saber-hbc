# Required Hydrological Datasets

1. Hindcast/Retrospective discharge for every stream segment (reporting point) in the model. This is a time series of
   discharge, e.g. hydrograph, for each stream segment. The data should be saved in parquet format and named 
   `hindcast_series_table.parquet`. The DataFrame should have:
    1. An index named `datetime` of type `datetime`. Contains the datetime stamp for the simulated values (rows)
    2. 1 column per stream, column name is the stream's model ID and is type string, containing the discharge for each
       time step.
2. Observed discharge data for each gauge. 1 file per gauge named `{gauge_id}.csv`. The DataFrame should have:
    1. `datetime`: The datetime stamp for the measurements
    2. A column whose name is the unique `gauge_id` containing the discharge for each time step.

The `hindcast_series_table.parquet` should look like this:

| datetime   | model_id_1 | model_id_2 | model_id_3 | ... |
|------------|------------|------------|------------|-----|
| 1985-01-01 | 50         | 50         | 50         | ... |
| 1985-01-02 | 60         | 60         | 60         | ... |
| 1985-01-03 | 70         | 70         | 70         | ... |
| ...        | ...        | ...        | ...        | ... |

Each gauge's csv file should look like this:

| datetime   | discharge |
|------------|-----------|
| 1985-01-01 | 50        |
| 1985-01-02 | 60        |
| 1985-01-03 | 70        |
| ...        | ...       |

## Things to check

Be sure that both datasets:

- Are in the same units (e.g. m3/s)
- Are in the same time zone (e.g. UTC)
- Are in the same time step (e.g. daily average)
- Do not contain any non-numeric values (e.g. ICE, none, etc.)
- Do not contain rows with missing values (e.g. NaN or blank cells)
- Have been cleaned of any incorrect values (e.g. no negative values)
- Do not contain any duplicate rows