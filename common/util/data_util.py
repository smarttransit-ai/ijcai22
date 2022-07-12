from osmnx import nearest_nodes

from base.common.MappedList import MappedList
from common.constant.constants import LOCATIONS_FILE_NAME, LOCATIONS_DATE_FILE_NAME, \
    TRAVEL_TIMES_DATE_FILE_NAME, TimeMatrixSource, TRAVEL_DISTANCES_DATE_FILE_NAME, TRAVEL_MATRICES_DATA_DIR
from common.util.common_util import read_data_file, get_agency_config, file_exists, logger
from common.util.pickle_util import dump_exists, load_obj, dump_obj


def obtain_locations(selected_agency, df):
    """
    :param df: pandas dataframe contains the para-transit trip data
    :param selected_agency: selected agency
    :return: list of distinct locations
    """
    locations = []
    round_off = get_agency_config(selected_agency).agency_round_off
    for i, entry in df.iterrows():
        pick_up_entry = (round(float(entry["Pickup lat"]), round_off), round(float(entry["Pickup lon"]), round_off))
        drop_off_entry = (round(float(entry["Dropoff lat"]), round_off), round(float(entry["Dropoff lon"]), round_off))

        if pick_up_entry not in locations:
            locations.append(pick_up_entry)

        if drop_off_entry not in locations:
            locations.append(drop_off_entry)
    return locations


def load_locations(
        selected_agency="AGENCY_A",
        selected_date=0,
        custom_file=None,
        date_filter_off=False,
        ignore_wc_runs=False
):
    """
    :param selected_agency: agency used to filter the para-transit trips
    :param selected_date: date used to filter the para-transit trips
    :param custom_file: custom file
    :param date_filter_off: turning off the date filter to read entire set of trips
    :param ignore_wc_runs: this enables filtering WC runs
    :return: list of distinct locations
    """
    if not date_filter_off:
        location_file = LOCATIONS_DATE_FILE_NAME.format(selected_agency, selected_date)
    else:
        location_file = LOCATIONS_FILE_NAME.format(selected_agency)
    if not dump_exists(location_file):
        df = read_data_file(
            selected_agency=selected_agency,
            selected_date=selected_date,
            custom_file=custom_file,
            date_filter_off=date_filter_off,
            ignore_wc_runs=ignore_wc_runs
        )
        locations = list(set(obtain_locations(selected_agency, df)))
        locations = [get_agency_config(selected_agency).depot_coordinates] + locations
        dump_obj(locations, location_file)
    else:
        locations = load_obj(location_file)
    return locations


def get_nearest_nodes_using_osmnx(graph, locations):
    """
    :param graph: NetworkX graph
    :param locations:  locations in the dataset
    :returns: the nearest OSM nodes for the given locations
    """
    nearest_locations = {}
    for i, (lat_i, lon_i) in enumerate(locations):
        nearest_node = nearest_nodes(graph, lon_i, lat_i)
        nearest_locations[(lat_i, lon_i)] = nearest_node
    return nearest_locations


def get_graph(selected_agency):
    """
    :param selected_agency: selected agency
    :returns: the nearest OSM nodes for the given locations
    """
    import osmnx as ox
    graph_file = "travel_time_matrix/data/osmnx_saved_graph"
    if file_exists(graph_file, ".graphml"):
        graph = ox.load_graphml(graph_file + ".graphml")
    else:
        area_location_name = get_agency_config(selected_agency).area_location_name
        ox.config(use_cache=True, log_console=False)
        graph = ox.graph_from_place(area_location_name, network_type='drive', simplify=False, retain_all=True)
    graph = ox.add_edge_speeds(graph, fallback=40.2, precision=6)
    return ox.add_edge_travel_times(graph, precision=6)


def load_nx_travel_times(selected_agency="AGENCY_A", selected_date=0):
    """
    This uses NetworkX, OSMnx travel-time
    :param selected_agency: selected agency
    :param selected_date:  selected date
    """
    travel_times_file = TRAVEL_TIMES_DATE_FILE_NAME.format(selected_agency, selected_date, TimeMatrixSource.NXN.value)
    if not dump_exists(travel_times_file):
        from datetime import datetime
        travel_times = {}
        count = 0
        missing = 0
        start_time = datetime.now()
        locations = load_locations(
            selected_agency=selected_agency, selected_date=selected_date
        )
        matrix, all_locations = load_nxn_main_dict_qa_tt(agency=selected_agency)

        all_locations = MappedList(all_locations)
        for i, location_i in enumerate(locations):
            for j, location_j in enumerate(locations):
                if i != j:
                    count += 1
                    try:
                        loc_i_idx = all_locations.index(location_i)
                        loc_j_idx = all_locations.index(location_j)
                        try:
                            travel_times[(i, j)] = matrix[(loc_i_idx, loc_j_idx)]
                        except KeyError:
                            missing += 1
                    except ValueError:
                        missing += 1

        dump_obj(travel_times, travel_times_file)
        end_time = datetime.now()
        if missing > 0:
            logger.warning(
                f"Travel Times Stats: Requested: {count}, Successful: {len(travel_times)}," +
                f" Failed: {missing}, Time taken: {(end_time - start_time).total_seconds()}(s)"
            )
    else:
        travel_times = load_obj(travel_times_file)
    return travel_times


def load_nx_travel_distance(selected_agency="AGENCY_A", selected_date=0):
    """
    :param selected_agency: selected agency
    :param selected_date:  selected date
    """
    travel_distances_file = TRAVEL_DISTANCES_DATE_FILE_NAME.format(
        selected_agency, selected_date, TimeMatrixSource.NXN.value
    )
    if not dump_exists(travel_distances_file):
        from datetime import datetime
        travel_distances = {}
        count = 0
        missing = 0
        start_time = datetime.now()
        locations = load_locations(
            selected_agency=selected_agency, selected_date=selected_date
        )
        matrix, all_locations = load_nxn_main_dict_qa_td(agency=selected_agency)

        all_locations = MappedList(all_locations)
        for i, location_i in enumerate(locations):
            for j, location_j in enumerate(locations):
                if i != j:
                    count += 1
                    loc_i_idx = all_locations.index(location_i)
                    loc_j_idx = all_locations.index(location_j)
                    try:
                        travel_distances[(i, j)] = matrix[(loc_i_idx, loc_j_idx)]
                    except KeyError:
                        missing += 1

        dump_obj(travel_distances, travel_distances_file)
        end_time = datetime.now()
        if missing > 0:
            logger.warning(
                f"Travel Distance Stats: Requested: {count}, Successful: {len(travel_distances)}," +
                f" Failed: {missing}, Time taken: {(end_time - start_time).total_seconds()}(s)"
            )
    else:
        travel_distances = load_obj(travel_distances_file)
    return travel_distances


def load_nxn_main_dict_qa_tt(agency="AGENCY_A"):
    travel_matrices_dir = TRAVEL_MATRICES_DATA_DIR.format(agency)
    matrix = load_obj(f"{travel_matrices_dir}/travel_times")
    locations = load_obj(f"{travel_matrices_dir}/locations")
    return matrix, locations


def load_nxn_main_dict_qa_td(agency="AGENCY_A"):
    travel_matrices_dir = TRAVEL_MATRICES_DATA_DIR.format(agency)
    matrix = load_obj(f"{travel_matrices_dir}/travel_distances")
    locations = load_obj(f"{travel_matrices_dir}/locations")
    return matrix, locations
