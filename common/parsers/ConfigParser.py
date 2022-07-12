import ast
from configparser import ConfigParser

from common.constant.constants import MIN_SEC_SWAP


class AgencyConfigParser(ConfigParser):
    _instance = None

    def __init__(self):
        super(AgencyConfigParser, self).__init__()

    @classmethod
    def config(cls, file_name):
        cls.__config_file_name = file_name
        cls._instance.read(cls.__config_file_name)
        cls.depot_coordinates = cls._instance.__literal_eval("AGENCY_CONF", "depot_coordinates")
        cls.main_transit_authority = cls._instance.__get_str("AGENCY_CONF", "main_transit_authority")
        cls.area_location_name = cls._instance.__get_str("AGENCY_CONF", "area_location_name")
        cls.agency_round_off = cls._instance.__get_int("AGENCY_CONF", "agency_round_off")
        cls.vehicle_capacity = cls._instance.__get_int("AGENCY_CONF", "vehicle_capacity")
        cls.run_max_duration = cls._instance.__get_int("AGENCY_CONF", "run_max_duration") * MIN_SEC_SWAP
        cls.run_weight = cls._instance.__get_int("AGENCY_CONF", "run_weight")
        cls.passenger_capacities = cls._instance.__literal_eval("AGENCY_CONF", "passenger_capacities")
        cls.wait_time = cls._instance.__get_int("AGENCY_CONF", "wait_time") * MIN_SEC_SWAP
        cls.time_window_gap = cls._instance.__get_int("AGENCY_CONF", "time_window_gap") * MIN_SEC_SWAP
        cls.broad_time_window_gap = cls._instance.__get_int("AGENCY_CONF", "broad_time_window_gap") * MIN_SEC_SWAP
        cls.dwell_time = cls._instance.__get_int("AGENCY_CONF", "dwell_time") * MIN_SEC_SWAP
        cls.drop_off_flexibility = cls._instance.__get_int("AGENCY_CONF", "drop_off_flexibility") * MIN_SEC_SWAP
        cls.max_detour_time = cls._instance.__get_int("AGENCY_CONF", "max_detour_time") * MIN_SEC_SWAP
        cls.planning_start_hour = cls._instance.__get_int("AGENCY_CONF", "planning_start_hour")
        cls.planning_end_hour = cls._instance.__get_int("AGENCY_CONF", "planning_end_hour")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = AgencyConfigParser()
        return cls._instance

    def __get_block(self, block_name):
        block = None
        try:
            block = self[block_name]
        except KeyError:
            print(f"Configuration block {block_name} not available in {self.__config_file_name}")
            exit(-1)
        return block

    def __get_elem_val(self, block_name, elem_name, data_type):
        elem = data_type(self.__get_block(block_name)[elem_name])
        return elem

    def __literal_eval(self, block_name, elem_name):
        elem = ast.literal_eval(self.__get_block(block_name)[elem_name])
        return elem

    def __get_int(self, block_name, elem_name):
        return self.__get_elem_val(block_name, elem_name, int)

    def __get_str(self, block_name, elem_name):
        return self.__get_elem_val(block_name, elem_name, str)
