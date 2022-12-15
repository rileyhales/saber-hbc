# Clustering
For each of the following, generate and store clusters for many group sizes- between 2 and 12 should be sufficient.
1. Create clusters of the *simulated* data by their flow duration curve.
2. Create clusters of the *simulated* data by their monthly averages.
3. Create clusters of the *observed* data by their flow duration curve.
4. Create clusters of the *observed* data by their monthly averages.
5. Track the error/inertia/residuals for each number of clusters identified.

This function creates trained kmeans models saved as pickle files, plots (from matplotlib) of what each of the clusters 
look like, and csv files which tracked the inertia (residuals) for each number of clusters. Use the elbow method to 
identify the correct number of clusters to use on each of the 4 datasets clustered.
