# Processing Input Data

Before following these steps, you should have prepared the required datasets and organized them in a working directory. 
Refer to the [Required Datasets](../data/index.md) page for more information.

***Prereqs:***

1. Create a working directory and subdirectories
2. Prepare the `drain_table` and `gauge_table` files.
3. Prepare the `hindcast_series_table` file.

## Prepare Flow Duration Curve Data

Process the `hindcast_series_table` to create a 2nd table with the flow duration curve on each segment.

| p_exceed | model_id_1 | model_id_2 | model_id_3 |
|----------|------------|------------|------------|
| 100      | 0          | 0          | 0          |
| 97.5     | 10         | 10         | 10         |
| 95       | 20         | 20         | 20         |
| ...      | ...        | ...        | ...        |

Then process the FDC data to create a 3rd table with scaled/transformed FDC data for each segment.

| model_id | Q100 | Q97.5 | Q95 |
|----------|------|-------|-----|
| 1        | 60   | 50    | 40  |
| 2        | 60   | 50    | 40  |
| 3        | 60   | 50    | 40  |
| ...      | ...  | ...   | ... |
