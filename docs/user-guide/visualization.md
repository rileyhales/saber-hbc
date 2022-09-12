### 9 Generate GIS files of the assignments
At any time during these steps you can use the functions in the `rbc.gis` module to create GeoJSON files which you can 
use to visualize the results of this process. These GIS files help you investigate which streams are being selected and 
used at each step. Use this to monitor the results.

```python
import saber as saber

workdir = '/path/to/project/directory/'
assign_table = saber.table.read(workdir)
drain_shape = '/my/file/path/'
saber.gis.clip_by_assignment(workdir, assign_table, drain_shape)
saber.gis.clip_by_cluster(workdir, assign_table, drain_shape)
saber.gis.clip_by_unassigned(workdir, assign_table, drain_shape)

# or if you have a specific set of ID's to check on
list_of_model_ids = [123, 456, 789]
saber.gis.clip_by_ids(workdir, list_of_model_ids, drain_shape)
```

After this step, your project directory should look like this:

