import os
import sys

import networkx as nx
import numpy as np
from pandas import DataFrame

sys.path.append(os.getcwd())
from common.parsers.ArgParser import CustomArgParser
from common.util.data_util import load_locations, get_graph, get_nearest_nodes_using_osmnx
from common.util.pickle_util import dump_obj, load_obj

travel_times = {}
travel_distances = {}

count = 0
missing = 0

arg_parser = CustomArgParser()
args = arg_parser.parse_args()

selected_agency = args.agency
locations = load_locations(
    selected_agency=selected_agency, date_filter_off=True, ignore_wc_runs=False
)

lats = [a for (a, b) in locations]
lons = [b for (a, b) in locations]
graph = get_graph(selected_agency=selected_agency)
nearest_nodes = get_nearest_nodes_using_osmnx(
    graph=graph, locations=locations
)
main_d_matrix = {}
main_t_matrix = {}

for i, node in enumerate(list(set(list(nearest_nodes.values())))):
    sub_d_matrix = nx.single_source_dijkstra_path_length(graph, source=node, weight='length')
    main_d_matrix[node] = sub_d_matrix
    sub_t_matrix = nx.single_source_dijkstra_path_length(graph, node, weight='travel_time')
    main_t_matrix[node] = sub_t_matrix
for i, location_i in enumerate(locations):
    for j, location_j in enumerate(locations):
        if i != j:
            count += 1
            nearest_node_i = nearest_nodes[locations[i]]
            nearest_node_j = nearest_nodes[locations[j]]
            try:
                travel_time = main_t_matrix[nearest_node_i][nearest_node_j]
                distance = main_d_matrix[nearest_node_i][nearest_node_j]
                if travel_time == np.infty or distance == np.infty:
                    raise KeyError
                travel_distances[(i, j)] = distance
                travel_times[(i, j)] = travel_time / 60
            except KeyError:
                missing += 1

dump_obj(travel_times, f"data/{selected_agency}/matrices/travel_times")
dump_obj(travel_distances, f"data/{selected_agency}/matrices/travel_distances")

travel_times = load_obj(f"data/{selected_agency}/matrices/travel_times")

keys = []
for i, loc_i in enumerate(locations):
    for j, loc_j in enumerate(locations):
        if loc_i != loc_j:
            keys.append((i, j))

missed_locations = []
while True:
    miss_count = {}
    missed_location = None
    for (i, j) in keys:
        loc_i = locations[i]
        loc_j = locations[j]
        if (i, j) not in travel_times.keys() and loc_i != loc_j:
            if str(loc_i) not in missed_locations and str(loc_j) not in missed_locations:
                if loc_i in miss_count.keys():
                    miss_count[loc_i] += 1
                else:
                    miss_count[loc_i] = 1
                if loc_j in miss_count.keys():
                    miss_count[loc_j] += 1
                else:
                    miss_count[loc_j] = 1
    if len(miss_count) > 0:
        for key in miss_count.keys():
            if miss_count[key] == max(miss_count.values()):
                missed_location = key
                break
        missed_locations.append(str(missed_location))
    else:
        break

df = DataFrame()
df["locations"] = missed_locations
df.to_csv(f"data/{selected_agency}/matrices/missed_locations.csv", index=False, sep=";")