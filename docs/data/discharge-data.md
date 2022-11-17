# Required Hydrological Datasets

- **Simulated Discharge** - hindcast discharge for every stream in Zarr format
- **Observed Discharge** - observed discharge for every gauge in CSV format
- **Simulated Discharge Flow Duration Curves** - hindcast discharge flow duration curves for every stream in Parquet format (calculated from the hindcast dataset)
- **Transformed Simulated FDC** - the flow duration curves transformed by the standard scalar and prepared for clustering in Parquet format (calculated from the FDC dataset)

## Simulated Discharge
Hindcast/Retrospective discharge for every stream segment (reporting point) in the model. This is a time series of
discharge, e.g. hydrograph, for each stream segment. The data should be saved in parquet format and named 
`hindcast_series_table.parquet`. The DataFrame should have:

1. An index named `datetime` of type `datetime`. Contains the datetime stamp for the simulated values (rows)
2. column per stream, column name is the stream's model ID and is type string, containing the discharge for each
       time step.

The table should look like this:

| datetime   | model_id_1 | model_id_2 | model_id_3 | ... |
|------------|------------|------------|------------|-----|
| 1985-01-01 | 50         | 50         | 50         | ... |
| 1985-01-02 | 60         | 60         | 60         | ... |
| 1985-01-03 | 70         | 70         | 70         | ... |
| ...        | ...        | ...        | ...        | ... |

## Observed Discharge
Observed discharge data for each gauge. 1 file per gauge named `{gauge_id}.csv`. The DataFrame should have:

1. `datetime`: The datetime stamp for the measurements
2. A column whose name is the unique `gauge_id` containing the discharge for each time step.

Each gauge's csv file should look like this:

| datetime   | discharge |
|------------|-----------|
| 1985-01-01 | 50        |
| 1985-01-02 | 60        |
| 1985-01-03 | 70        |
| ...        | ...       |

## Transformed Simulated FDC
The flow duration curves transformed by the standard scalar and prepared for clustering in Parquet format. The FDC should 
be calculated as exceedance probabilities (1 - nonexceedance probability) at 5% intervals, including 0% and 100% for a total 
of 21 points (columns in the table). Each FDC should be saved as 1 row in a parquet file named `hindcast_fdc_transformed.parquet`.

- Calculate exceedance probabilities in 5% intervals, including 0% and 100%, for a total of 21 points
- Do not include a column of Model IDs. The label for each row should be saved separately in a file named `model_id_list.parquet`
- Apply the standard scalar to each stream independently. Do not calculate a mean and standard deviation considering all points.

It should look like this

The values typically vary between approximately 4 and -2. An example table with sample rows between those values might look like this:

| fdc_0 | fdc_5 | ... | fdc_95 | fdc_100 |
|-------|-------|-----|--------|---------|
| 4     | 3.8   | ... | -1.8   | -2      |
| 4     | 3.8   | ... | -1.8   | -2      |
| 4     | 3.8   | ... | -1.8   | -2      |


## Things to check

Be sure that both datasets:

- Are in the same units (e.g. m3/s)
- Are in the same time zone (e.g. UTC)
- Are in the same time step (e.g. daily average)
- Do not contain any non-numeric values (e.g. ICE, none, etc.)
- Do not contain rows with missing values (e.g. NaN or blank cells)
- Have been cleaned of any incorrect values (e.g. no negative values)
- Do not contain any duplicate rows