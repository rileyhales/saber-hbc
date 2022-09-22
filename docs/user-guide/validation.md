## Analyzing Performance
This bias correction method should adjust all streams toward a more realistic but still not perfect value for discharge. 
The following steps help you to analyze how well this method performed in a given area by iteratively running this 
method on the same set of streams but with an increasing number of randomly selected observed data stations being 
excluded each time. The code provided will help you partition your gauge table into randomly selected subsets.

### Steps
1. Perform the bias correction method with all available observed data.
2. Generate 5 subsets of the gauge table (using provided code)
   1. One with ~90% of the gauges (drop a random 10% of the observed data stations)
   2. One with ~80% of the gauges (drop the same gauges as before ***and*** and additional random 10%)
   3. One with ~70% of the gauges (drop the same gauges as before ***and*** and additional random 10%)
   4. One with ~60% of the gauges (drop the same gauges as before ***and*** and additional random 10%)
   5. One with ~50% of the gauges (drop the same gauges as before ***and*** and additional random 10%)
3. Perform the bias correction method 5 additional times using the 5 new tables created in the previous step. You now
   have 6 separate bias correction instances; 1 with all available observed data and 5 with decreasing amounts of 
   observed data included.
4. For each of the 5 corrected models with observed data withheld, use the provided code to generate plots and maps of 
   the performance metrics. This will compare the best approximation of the bias corrected model data for that instance 
   against the observed data which was withheld from the bias correction process.

```python
import saber as saber
workdir = '/path/to/project/directory'
drain_shape = '/path/to/drainageline/gis/file.shp'
obs_data_dir = '/path/to/obs/data/directory'  # optional - if data not in workdir/data_inputs/obs_csvs

saber.validate.sample_gauges(workdir)
saber.validate.run_series(workdir, drain_shape, obs_data_dir)
```
