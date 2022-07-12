import math
import random

import pandas as pd
from pandas import DataFrame

from base.common.MappedList import MappedList
from base.common.TravelDistanceMatrix import TravelDistanceMatrix
from base.common.TravelTimeMatrix import TravelTimeMatrix
from common.constant.constants import MIN_SEC_SWAP, \
    GENERIC_DATA_FILE_NAME, ALGO_DATA_FILE_NAME, GENERIC_DATA_DIR, SHORTEN_DATA_FILE_NAME, VEHICLE_INFO_FILE_NAME, \
    DAY_IN_MINUTES, CSV_EXTENSION, MISSED_LOCATIONS_FILE_NAME
from common.util.common_util import get_time_stamp, read_data_file, create_dir, format_time, logger, get_agency_config, \
    file_exists
from common.util.data_util import load_locations
from common.util.pickle_util import dump_obj


class GenericData(object):
    """
        This class stores generic data from para-transit dataset
    """

    def __init__(
            self,
            sample_inst_config=None,
            config=None,
            custom_agency_config=None
    ):
        """
        :param sample_inst_config: configuration for sample instances
        :param config: configurations for data processing
        :param custom_agency_config: custom agency configuration
        """
        self.sample_inst_config = sample_inst_config
        self.config = config
        if custom_agency_config is None:
            self.agency_config = get_agency_config(sample_inst_config.selected_agency)
        else:
            self.agency_config = custom_agency_config

        # distance matrices
        self.__travel_time_matrix = None
        self.__travel_distance_matrix = None

        # array to store pickup and drop_off nodes
        self.pick_up_nodes = []
        self.drop_off_nodes = []

        # array to store pickup and drop_off time_time_windows (just for ease of access)
        self.travel_times = []
        self.pick_up_time_windows = []
        self.drop_off_time_windows = []
        self.pick_up_broad_time_windows = []
        self.capacities = []
        self.distinct_locations = []
        self.wheel_chair_counts = []
        self.client_ids = []
        self.booking_ids = []
        self.scheduled_times = []
        self.pick_up_zips = []
        self.drop_off_zips = []
        self.accepted_ids = []
        self.actual_number_of_trips = 0
        self.true_locations = {0: self.agency_config.depot_coordinates}
        self._load()

    def _load(self):
        df = read_data_file(
            selected_agency=self.sample_inst_config.selected_agency,
            selected_date=self.sample_inst_config.selected_date,
        )
        logger.info(f"Loading data, for AGENCY:  {self.sample_inst_config.selected_agency},"
                    f" DATE: {self.sample_inst_config.selected_date}")

        if len(df) == 0:
            logger.warning(f"No request found for date {self.sample_inst_config.selected_date}")
            exit()

        logger.info(f"Loading un-filtered data, total number of requests: {len(df)}")

        # load locations
        self.__locations = MappedList(load_locations(
            selected_agency=self.sample_inst_config.selected_agency,
            selected_date=self.sample_inst_config.selected_date,
        ))
        logger.info(f"Locations Loaded, number of locations: {self.__locations.size()}")

        travel_time_expected = self.__locations.size() * (self.__locations.size() - 1)

        # load travel time data
        self.__travel_time_matrix = TravelTimeMatrix(
            travel_time_data_set=self.config.travel_time_data_set,
            selected_agency=self.sample_inst_config.selected_agency,
            selected_date=self.sample_inst_config.selected_date,
        )

        miss_percentage = (travel_time_expected - self.__travel_time_matrix.size()) * 100 / travel_time_expected
        if miss_percentage > 0:
            logger.warning(f"Travel Times Loaded, missing of travel-time entries {round(miss_percentage, 3)}%")

        self.__travel_distance_matrix = TravelDistanceMatrix(
            selected_agency=self.sample_inst_config.selected_agency,
            selected_date=self.sample_inst_config.selected_date,
        )

        missed_file_name = MISSED_LOCATIONS_FILE_NAME.format(
            self.sample_inst_config.selected_agency,
            self.sample_inst_config.selected_date,
        )

        if file_exists(missed_file_name, CSV_EXTENSION):
            df_missed = pd.read_csv(missed_file_name + CSV_EXTENSION, delimiter=";")
            missed_locations = df_missed["locations"].to_list()
        else:
            missed_locations = self.get_missed_locations(file_name=missed_file_name + CSV_EXTENSION)

        if missed_locations is not None:
            self.accepted_ids = []
            for i in range(len(df)):
                entry = df.iloc[i]

                # obtain the pick-up coordinates
                pick_up_entry = (round(entry["Pickup lat"], self.agency_config.agency_round_off),
                                 round(entry["Pickup lon"], self.agency_config.agency_round_off))

                # obtain the drop-off coordinates
                drop_off_entry = (round(entry["Dropoff lat"], self.agency_config.agency_round_off),
                                  round(entry["Dropoff lon"], self.agency_config.agency_round_off))

                if not (str(pick_up_entry) in missed_locations or str(drop_off_entry) in missed_locations):
                    self.accepted_ids.append(i)
        else:
            self.accepted_ids = [i for i in range(len(df))]

        if len(self.accepted_ids) != len(df):
            logger.warning(f"Obtained the filtered request, total number of filtered request: {len(self.accepted_ids)}")

        # computing the start time for selected date
        stmp = get_time_stamp(self.sample_inst_config.selected_date)

        if self.sample_inst_config.no_of_trips == -1:
            # use all the trips
            self.actual_number_of_trips = len(self.accepted_ids)
        else:
            self.actual_number_of_trips = min(self.sample_inst_config.no_of_trips, len(self.accepted_ids))

        # need to check the random-ness
        random.seed(self.sample_inst_config.random_seed)
        self.accepted_ids = random.sample(self.accepted_ids, self.actual_number_of_trips)
        passenger_count = 0

        for i in range(self.actual_number_of_trips):
            # iterate over the data
            entry = df.iloc[self.accepted_ids[i]]

            # obtain the pick-up coordinates
            pick_up_entry = (round(entry["Pickup lat"], self.agency_config.agency_round_off),
                             round(entry["Pickup lon"], self.agency_config.agency_round_off))

            # obtain the drop-off coordinates
            drop_off_entry = (round(entry["Dropoff lat"], self.agency_config.agency_round_off),
                              round(entry["Dropoff lon"], self.agency_config.agency_round_off))

            missing, travel_time = self.get_travel_time(pick_up_entry, drop_off_entry)

            time_string = entry["Sch Time in HH:MM:SS"]
            date_string = entry["Date"]

            # obtain the start timestamp (actually earliest pick-up time, where latest pick-up time can be
            # 30 minutes from earliest pickup time)
            start_timestamp = get_time_stamp(date_string, time_string)
            start_timestamp = int((start_timestamp - stmp) * MIN_SEC_SWAP / 60)

            # computing the end time stamp (actually an approximate estimate the earliest possible drop-off time,
            end_timestamp = start_timestamp + travel_time
            pickup_tw = (start_timestamp - int(self.agency_config.time_window_gap / 2),
                         start_timestamp + int(self.agency_config.time_window_gap / 2))
            pickup_b_tw = (start_timestamp - int(self.agency_config.broad_time_window_gap / 2),
                           start_timestamp + int(self.agency_config.broad_time_window_gap / 2))
            dropoff_tw = (end_timestamp - int(self.agency_config.time_window_gap / 2),
                          end_timestamp + int(self.agency_config.time_window_gap / 2))
            if self.config.exact_passenger_count:
                capacity, passenger_count_dict = self.get_passenger_capacity(entry)
                wheel_chair_count_dict = {}
                for _key in passenger_count_dict:
                    if _key.startswith("W"):
                        wheel_chair_count_dict[_key] = passenger_count_dict[_key]
                passenger_count += sum(passenger_count_dict.values())
                self.capacities.append(capacity)
                self.wheel_chair_counts.append(wheel_chair_count_dict)
            else:
                self.capacities.append(1.0)
            self.travel_times.append(travel_time)
            self.pick_up_nodes.append(2 * i + 1)
            self.pick_up_time_windows.append(pickup_tw)
            self.pick_up_broad_time_windows.append(pickup_b_tw)
            self.true_locations[2 * i + 1] = pick_up_entry
            self.drop_off_nodes.append(2 * i + 2)
            self.drop_off_time_windows.append(dropoff_tw)
            self.true_locations[2 * i + 2] = drop_off_entry
            # these are just added for some printing purposes
            client_id = -1
            if "Client Id" in entry.keys():
                client_id = entry["Client Id"]
            booking_id = -1
            if "Booking Id" in entry.keys():
                booking_id = entry["Booking Id"]
            if str(entry['Pickup Zip']) == 'nan':
                self.pick_up_zips.append(-1)
            else:
                self.pick_up_zips.append(int(entry["Pickup Zip"]))
            if str(entry["Dropoff Zip"]) == 'nan':
                self.drop_off_zips.append(-1)
            else:
                self.drop_off_zips.append(int(entry["Dropoff Zip"]))
            self.client_ids.append(client_id)
            self.booking_ids.append(booking_id)
            self.scheduled_times.append(time_string)

        logger.info(f"Generated the data, total number of request in data: {self.actual_number_of_trips}")

        if self.config.exact_passenger_count:
            self.total_passengers = passenger_count
        else:
            self.total_passengers = len(self.accepted_ids)

        self.distinct_locations = list(set(self.true_locations.values()))

        # create dummy set of vehicles
        self.vehicles = [f"v_{v}" for v in range(self.sample_inst_config.no_of_vehicles)]
        self.no_of_vehicles = self.sample_inst_config.no_of_vehicles
        self._save_readable(
            data=df,
            prefixes=(
                self.sample_inst_config.selected_agency,
                self.sample_inst_config.selected_date,
                self.sample_inst_config.no_of_trips,
                self.sample_inst_config.no_of_runs,
                self.sample_inst_config.no_of_vehicles,
                self.sample_inst_config.random_seed
            )
        )
        logger.info(f"Save readable data, total number of request in data: {self.actual_number_of_trips}")

    def get_data_size(self):
        """
        :return: returns the actual problem size
        """
        return self.actual_number_of_trips

    def get_slot_per_pole(self):
        """
        :return: returns the number of charging slot per pole
        """
        return int(DAY_IN_MINUTES / self.sample_inst_config.charging_duration)

    def get_charge_size(self):
        """
        :return: returns the charging slot count
        """
        return self.sample_inst_config.no_of_charging_poles * self.get_slot_per_pole()

    def get_passenger_capacity(self, entry):
        """
        :param entry: the dataframe entry, which contains the passenger details
        :return: total passenger capacity, and passenger count dictionary
        """
        passenger_types = str(entry["Passenger Types"])

        def find_count(val_str, prefix):
            """
            :param val_str: passenger count values
            :param prefix: passenger type prefix
            :return: number of passenger available for particular passenger types
            """
            val = 0
            if prefix in val_str:
                val = val_str.replace(prefix, "")
            return int(val)

        def get_passenger_count(prefix):
            val = 0
            if "," in passenger_types:
                val_subs = passenger_types.split(",")
                for val_sub in val_subs:
                    val = find_count(val_sub, prefix)
                    if val != 0:
                        break
            else:
                val = find_count(passenger_types, prefix)
            return val

        capacity = 0.0
        passenger_count_dict = {}
        for key in self.agency_config.passenger_capacities.keys():
            count = get_passenger_count(key)
            capacity += self.agency_config.passenger_capacities[key] * count
            passenger_count_dict[key] = count
        return capacity, passenger_count_dict

    def get_travel_time(self, _loc_i, _loc_j):
        """
        :param _loc_i: start location
        :param _loc_j: end location
        :return: travel time (IN SECONDS, OR IN MINUTES IF MIN_SEC_SWAP = 1) and missing status
        """
        if _loc_i == _loc_j:
            travel_time = 0
            missing = False
        else:
            _loc_i_idx = self.__locations.index(_loc_i)
            _loc_j_idx = self.__locations.index(_loc_j)
            missing, travel_time = self.__travel_time_matrix.get_travel_time(_loc_i_idx, _loc_j_idx)
        if travel_time != math.inf:
            travel_time = int(travel_time * MIN_SEC_SWAP)
        return missing, travel_time

    def get_distance(self, _loc_i, _loc_j):
        """
        :param _loc_i: start location
        :param _loc_j: end location
        :return: travel time (IN SECONDS, OR IN MINUTES IF MIN_SEC_SWAP = 1) and missing status
        """
        if _loc_i == _loc_j:
            distance = 0
            missing = False
        else:
            _loc_i_idx = self.__locations.index(_loc_i)
            _loc_j_idx = self.__locations.index(_loc_j)
            missing, distance = self.__travel_distance_matrix.get_distance(_loc_i_idx, _loc_j_idx)
            if distance != math.inf:
                distance = int(distance)
        return missing, distance

    def get_travel_time_by_idx(self, _i, _j):
        """
        :param _i: start location index in self.true_locations
        :param _j: end location index in self.true_locations
        :return: travel time and missing status
        """
        _loc_i = self.true_locations[_i]
        _loc_j = self.true_locations[_j]
        # to avoid floating point errors
        if _i > 0:
            round_val_i = self.agency_config.agency_round_off
        else:
            round_val_i = 14
        _loc_i_la = round(_loc_i[0], round_val_i)
        _loc_i_lo = round(_loc_i[1], round_val_i)
        _loc_i = (_loc_i_la, _loc_i_lo)
        if _j > 0:
            round_val_j = self.agency_config.agency_round_off
        else:
            round_val_j = 14
        _loc_j_la = round(_loc_j[0], round_val_j)
        _loc_j_lo = round(_loc_j[1], round_val_j)
        _loc_j = (_loc_j_la, _loc_j_lo)
        return self.get_travel_time(_loc_i, _loc_j)

    def get_time_matrix(self):
        """
        :return: returns travel time matrix
        """
        data = {
            'time_matrix': [
                [0 for _ in range(self.get_data_size() * 2 + 1)]
                for _ in range(self.get_data_size() * 2 + 1)
            ]
        }
        miss_count = 0
        for i in range(0, 2 * self.get_data_size() + 1):
            for j in range(0, 2 * self.get_data_size() + 1):
                if self.true_locations[i] != self.true_locations[j]:
                    missing, travel_time = self.get_travel_time_by_idx(i, j)
                    if missing:
                        miss_count += 1
                else:
                    travel_time = 0
                data['time_matrix'][i][j] = int(travel_time)
        total = (2 * self.get_data_size() + 1) * (2 * self.get_data_size())
        percentage_missing = round(miss_count * 100.0 / total, 3)
        if percentage_missing != 0:
            logger.warning(f"Travel Time Missing {percentage_missing}%")
        return data['time_matrix'].copy()

    def _dump_data(self, dump_file_name=GENERIC_DATA_FILE_NAME, prefixes=None):
        """
        :param dump_file_name: file name for dumping the generic data for the solver
        :param prefixes: prefixes to identify the specific dump, based on number of trips, number of vehicles
        """
        dump_obj(self, dump_file_name, prefixes)

    def _dump_data_model(self, dump_file_name=ALGO_DATA_FILE_NAME, prefixes=None):
        """
        :param dump_file_name: file name for dumping the algo specific data for the solver
        :param prefixes: prefixes to identify the specific dump, based on algorithm, number of trips, number of vehicles
        """
        dump_obj(self.get_data_model(), dump_file_name, prefixes)

    def get_data_model(self):
        raise NotImplementedError

    def save_readable(self, df=None, sample_inst_config=None):
        """
        :param df:  data frame
        :param sample_inst_config: custom sample instance configuration
        """
        if sample_inst_config is None:
            sample_inst_config = self.sample_inst_config

        if df is None:
            df = read_data_file(
                selected_agency=sample_inst_config.selected_agency,
                selected_date=sample_inst_config.selected_date,
            )

        self._save_readable(
            data=df,
            prefixes=(
                sample_inst_config.selected_agency,
                sample_inst_config.selected_date,
                sample_inst_config.no_of_trips,
                sample_inst_config.no_of_runs,
                sample_inst_config.no_of_vehicles,
                sample_inst_config.random_seed
            )
        )

    def _save_readable(
            self,
            data,
            gen_data_directory=GENERIC_DATA_DIR,
            prefixes=None
    ):
        """
        :param data:  data frame
        :param gen_data_directory: folder name for storing generic data
        :param prefixes: prefixes to identify the specific dump, based on number of trips, number of vehicles
        """
        gen_data_directory = gen_data_directory.format(*prefixes)
        create_dir(gen_data_directory)
        data_short = data.iloc[self.accepted_ids]
        pd.options.mode.chained_assignment = None
        data_short["Earliest Pickup"] = [format_time(a) for (a, _) in self.pick_up_time_windows]
        data_short["Latest Pickup"] = [format_time(b) for (_, b) in self.pick_up_time_windows]
        data_short["Earliest Dropoff"] = [format_time(a) for (a, _) in self.drop_off_time_windows]
        data_short["Latest Dropoff"] = [format_time(b) for (_, b) in self.drop_off_time_windows]
        data_short["Travel Time (min)"] = [travel_time / 60 for travel_time in self.travel_times]
        data_short.to_csv(SHORTEN_DATA_FILE_NAME.format(*prefixes), index=False)
        pd.options.mode.chained_assignment = 'warn'

        if len(self.vehicles) > 0:
            vehicle_info = DataFrame()
            vehicle_info["Vehicle ID"] = self.vehicles
            vehicle_info["Capacity"] = [self.agency_config.vehicle_capacity] * len(self.vehicles)
            vehicle_info["Start Location"] = [self.agency_config.depot_coordinates] * len(self.vehicles)
            vehicle_info.to_csv(VEHICLE_INFO_FILE_NAME.format(*prefixes), index=False)
        else:
            logger.warning("Vehicle info file not generated")

    def _save_readable_wrap(self):
        """
            save readable if the system insert request one by one
        """
        self.accepted_ids = [k for k in range(len(self.pick_up_nodes))]
        df = DataFrame()
        df["Booking Id"] = self.booking_ids
        df["Client Id"] = self.client_ids
        self._save_readable(
            data=df,
            prefixes=(
                self.sample_inst_config.selected_agency,
                self.sample_inst_config.selected_date,
                self.sample_inst_config.no_of_trips,
                self.sample_inst_config.no_of_runs,
                self.sample_inst_config.no_of_vehicles,
                self.sample_inst_config.random_seed
            )
        )

    def get_missed_locations(self, file_name):
        """
        :param file_name: name of the file which store the missed locations
        :return: the locations which has missing entries in travel-time matrix
        """
        keys = []
        for i, loc_i in enumerate(self.__locations.get_list()):
            for j, loc_j in enumerate(self.__locations.get_list()):
                if loc_i != loc_j:
                    keys.append((i, j))

        missed_locations = []
        while True:
            miss_count = {}
            missed_location = None
            for (i, j) in keys:
                loc_i = self.__locations.get_list()[i]
                loc_j = self.__locations.get_list()[j]
                if (i, j) not in self.__travel_time_matrix.keys() and loc_i != loc_j:
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
        df.to_csv(file_name, index=False, sep=";")
        return missed_locations.copy()
