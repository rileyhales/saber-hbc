# hydrobias

## Steps
1. Prepare the data
    * Shapefile/GeoJSON (lines) of the drainagelines of the model with an attribute specifying the segment's ID
    * Shapefile/GeoJSON (points) of the location of the measurement stations. These should have already been post processed to overlap the drainageline network using snapping if applicable.
    * For faster access, create a csv of the attribute table of both of these. We save time by operating on the tables rather than the spatial geometry.
2. Create clusters of the simulated data. Create clusters of the observed data.
3. Spatially identify paired clusters of observed data and simulated data.
4. Assign ungauged basins the ID of observed data for use via the propagation method.
5. Assign ungauged basins the ID of observed data for use via the clustering similarities.
6. Identify ungauged basins that were not assigned observed data for corrections.
7. Review assignments spatially. Run tests and view improvements. Adjust clusters and reassign as necessary.
8. Export the resulting csv of assignments.
9. Use the csv to guide applying the correction scripts in various applications.