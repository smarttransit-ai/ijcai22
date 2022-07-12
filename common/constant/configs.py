from common.constant.constants import TimeMatrixSource


class SampleInstanceConfiguration(object):
    def __init__(
            self,
            selected_agency="AGENCY_A",
            selected_date=0,
            random_seed=0,
            no_of_trips=-1,
            no_of_vehicles=-1,
            no_of_runs=-1,
            vehicle_capacity=-1,
            dwell_time=-1,
            drop_off_flexibility=-1,
            broad_time_window_gap=-1
    ):
        """
        :param selected_agency: name of the agency used to filter the para-transit trips
        :param selected_date: date used to filter the para-transit trips
        :param random_seed: indicates the random seed
        :param no_of_trips: number of trips for the solving the problem
        :param no_of_runs: number of runs for the solving the problem
        :param no_of_vehicles: number of vehicles for the solving the problem
        :param vehicle_capacity: vehicle capacity
        :param dwell_time: dwell time
        :param drop_off_flexibility: drop-off flexibility
        :param broad_time_window_gap: broad time window gap
        """
        self.selected_agency = selected_agency
        self.selected_date = selected_date
        self.random_seed = random_seed
        self.no_of_trips = no_of_trips
        self.no_of_runs = no_of_runs
        self.no_of_vehicles = no_of_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.dwell_time = dwell_time
        self.drop_off_flexibility = drop_off_flexibility
        self.broad_time_window_gap = broad_time_window_gap

    def copy(self):
        sample_inst_copy = SampleInstanceConfiguration()
        sample_inst_copy.selected_agency = self.selected_agency
        sample_inst_copy.selected_date = self.selected_date
        sample_inst_copy.random_seed = self.random_seed
        sample_inst_copy.no_of_trips = self.no_of_trips
        sample_inst_copy.no_of_runs = self.no_of_runs
        sample_inst_copy.no_of_vehicles = self.no_of_vehicles
        sample_inst_copy.vehicle_capacity = self.vehicle_capacity
        sample_inst_copy.dwell_time = self.dwell_time
        sample_inst_copy.drop_off_flexibility = self.drop_off_flexibility
        sample_inst_copy.broad_time_window_gap = self.broad_time_window_gap
        return sample_inst_copy


class DataConfiguration(object):
    def __init__(
            self,
            exact_passenger_count=True,
            skip_dump_process=False,
            travel_time_data_set=TimeMatrixSource.NXN
    ):
        """
        :param exact_passenger_count: if true make exact passenger count
        :param skip_dump_process: skip dump generation process
        :param travel_time_data_set: define the travel time dataset
        """
        self.exact_passenger_count = exact_passenger_count
        self.skip_dump_process = skip_dump_process
        self.travel_time_data_set = travel_time_data_set
