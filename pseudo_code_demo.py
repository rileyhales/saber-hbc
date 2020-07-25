def assign_id_for_correction():
    # make the csv that looks like this
    # comid, station id, assigned id
    # start with a list of all comids, merge that df with the df showing the comid and station id pairs, sort it

    # if station id is not nan, then assigned id is the station id

    # for each station id we have
        # find the comid
        # get the next down id

        # case 1
        # does this next down segment have a gauge? if yes

        # is the stream order the same?
        #

        # does that next down id have multiple upstream id's?


    clip the drainagelines to the cluster of interest
    for each stream:
        create a geojson of the outlet point of the stream

    on each gauge location
    walk up and down the stream until you find a higher stream order or the

    filter the catchments and station data
    for each combination of spatially intersecting station and catchment
        make voroni polygons around the stations with the extents being the boundary of the watershed
        select the catchments using the location of the centroid and the voroni polygons
        the catchments who are within the voroni polygons get assigned the station number of the polygon it is in
