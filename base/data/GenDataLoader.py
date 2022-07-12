##############################
#     GENERIC DATA LOADER    #
##############################
import datetime
import math

from base.data.CSData import CSData
from base.features.DemandPredictor import DemandPredictor
from base.features.ExpectedReqGenerator import ExpectedReqGenerator
from common.constant.constants import MIN_SEC_SWAP
from common.util.common_util import get_agency_config


class GenDataLoader(object):
    """
        This class should be initialized only once, and load data multiple times
        and accessible throughout the system, without storing in the individual objects
    """
    _instance = None

    @classmethod
    def load(cls, gen_data_cls=CSData, args=None):
        sample_inst_config, config, agency_config = args
        if sample_inst_config is not None:
            if agency_config is None:
                agency_config = get_agency_config(sample_inst_config.selected_agency)
            if sample_inst_config.vehicle_capacity > 0:
                agency_config.vehicle_capacity = sample_inst_config.vehicle_capacity
            if sample_inst_config.dwell_time >= 0:
                agency_config.dwell_time = sample_inst_config.dwell_time * MIN_SEC_SWAP
            if sample_inst_config.drop_off_flexibility >= 0:
                agency_config.drop_off_flexibility = sample_inst_config.drop_off_flexibility * MIN_SEC_SWAP
            if sample_inst_config.broad_time_window_gap >= 0:
                agency_config.broad_time_window_gap = sample_inst_config.broad_time_window_gap * MIN_SEC_SWAP
        cls.agency = sample_inst_config.selected_agency
        cls.date = sample_inst_config.selected_date
        cls.day_of_week = int(cls._instance.date) % 7
        cls.gen_data = gen_data_cls(sample_inst_config, config, agency_config)
        cls.data = cls._instance.gen_data.get_data_model()
        cls.agency_config = agency_config
        cls.exp_requests = ExpectedReqGenerator(cls._instance.agency)
        cls.demand_predictor = DemandPredictor(cls._instance.agency)

    def get_travel_time(self, loc_i, loc_j):
        """
        this provides a quick function call rather than calling through GenericData object
        :param loc_i: start location
        :param loc_j: end location
        :return: the travel-time
        """
        return self._instance.gen_data.get_travel_time(loc_i, loc_j)

    def get_distance(self, loc_i, loc_j):
        """
        this provides a quick function call rather than calling through GenericData object
        :param loc_i: start location
        :param loc_j: end location
        :return: the distance
        """
        return self._instance.gen_data.get_distance(loc_i, loc_j)

    def get_data_size(self):
        """
        :return: the size of the data
        """
        return self._instance.gen_data.get_data_size()

    def exp_req(self, h, dow=None):
        """
        :return: the expected number of requests at the given time of the day and day of the week
        """
        if dow is None:
            dow = self._instance.day_of_week
        exp_request = self._instance.exp_requests.expected_req[dow]
        start = self._instance.agency_config.planning_start_hour
        end = self._instance.agency_config.planning_end_hour
        return exp_request * (end - h) * 1.0 / (end - start)

    def exp_req_by_request(self, request, dow=None):
        """
        :return: the expected number of requests at the given time of the day and day of the week
        """
        if dow is None:
            dow = self._instance.day_of_week
        h = int(math.floor(request.pick_up_node.earliest_arrival / (MIN_SEC_SWAP * 60)))
        return self._instance.exp_req(h, dow)

    def demand(self, pz=-1, dz=-1, h=-1, dow=None):
        """
        :return: the demand for given pickup zip, drop-off zip, time of day and day of the week
        """
        if dow is None:
            dow = self._instance.day_of_week
        demand = 0
        # when all pick-up zip, drop-off zip, time of day and day of the week
        # are provided
        if pz != -1 and dz != -1 and h != -1:
            if (pz, dz, h, dow) in self._instance.demand_predictor.pdh_demand_dict.keys():
                demand = self._instance.demand_predictor.pdh_demand_dict[(pz, dz, h, dow)]

        # when only pick-up zip, time of day and day of the week
        # are provided
        elif pz != -1 and dz == -1 and h != -1:
            if (pz, h, dow) in self._instance.demand_predictor.ph_demand_dict.keys():
                demand = self._instance.demand_predictor.ph_demand_dict[(pz, h, dow)]

        # when only drop-off zip, time of day and day of the week
        # are provided
        elif pz == -1 and dz != -1 and h != -1:
            if (dz, h, dow) in self._instance.demand_predictor.dh_demand_dict.keys():
                demand = self._instance.demand_predictor.dh_demand_dict[(dz, h, dow)]

        # when time of day and day of the week are provided
        elif pz == -1 and dz == -1 and h != -1:
            if (h, dow) in self._instance.demand_predictor.h_demand_dict.keys():
                demand = self._instance.demand_predictor.h_demand_dict[(h, dow)]
        return demand

    def demand_by_request(self, request, dow=None):
        """
        :return: the demand for given pickup zip, drop-off zip, time of day and day of the week
        based on the request
        """
        if dow is None:
            dow = self._instance.day_of_week
        pz = request.pick_up_zip
        dz = request.drop_off_zip
        h = int(math.floor(request.pick_up_node.earliest_arrival / (MIN_SEC_SWAP * 60)))
        return self._instance.demand(pz, dz, h, dow)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = GenDataLoader()
        return cls._instance
