# script from user ctvrtkar https://www.reddit.com/r/gis/comments/b1ui7h/geopandas_how_to_make_a_graph_out_of_a/
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point


def gdf_to_nx(gdf_network):
    # generate graph from GeoDataFrame of LineStrings
    net = nx.Graph()
    net.graph['crs'] = gdf_network.crs
    fields = list(gdf_network.columns)

    for index, row in gdf_network.iterrows():
        first = row.geometry.coords[0]
        last = row.geometry.coords[-1]

        data = [row[f] for f in fields]
        attributes = dict(zip(fields, data))
        net.add_edge(first, last, **attributes)

    return net


def nx_to_gdf(net, nodes=True, edges=True):
    # generate nodes and edges geodataframes from graph
    if nodes is True:
        node_xy, node_data = zip(*net.nodes(data=True))
        gdf_nodes = gpd.GeoDataFrame(list(node_data), geometry=[Point(i, j) for i, j in node_xy])
        gdf_nodes.crs = net.graph['crs']

    if edges is True:
        starts, ends, edge_data = zip(*net.edges(data=True))
        gdf_edges = gpd.GeoDataFrame(list(edge_data))
        gdf_edges.crs = net.graph['crs']

    if nodes is True and edges is True:
        return gdf_nodes, gdf_edges
    elif nodes is True and edges is False:
        return gdf_nodes
    else:
        return gdf_edges

plt.figure(figsize=(6, 6))
a = gpd.read_file('../data_0_inputs/magdalena_single_lines.geojson')
a = a[a['order_'] == 5]
b = gdf_to_nx(a)
print(b.nodes)
print(b)
plt
# plt.subplot(121)
nx.draw(b)
plt.show()
