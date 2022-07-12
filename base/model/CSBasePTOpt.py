import random
import sys

from base.data.CSData import CSData
from base.data.GenDataLoader import GenDataLoader
from base.entity.Node import Node
from base.entity.Request import Request
from base.entity.Vehicle import Vehicle
from base.errors import ImplementationError
from common.constant.constants import GENERIC_DATA_FILE_NAME, SHORTEN_DATA_FILE_NAME, \
    AGENCY_ASSIGNS_DIR, AGENCY_PLANS_DIR, AGENCY_IMAGES_DIR, DAY_IN_MINUTES, MIN_SEC_SWAP
from common.util.common_util import create_dir, convert_sec_to_hh_mm_ss, visualize, logger
from common.util.pickle_util import dump_exists, load_obj


class CSBasePTOpt(object):
    """
        This class stores generic solver setup to solve para-transit optimization problem,
        using custom solver
    """

    def __init__(
            self,
            prefix,
            sample_inst_config=None,
            config=None,
            custom_args=None,
            booking_ids=None
    ):
        self.prefix = prefix
        self.sample_inst_config = sample_inst_config
        self.config = config
        summary_file_common = self.get_summary_suffix()
        self.summary_file_name = f"{AGENCY_PLANS_DIR}/{summary_file_common}.txt"
        self.summary_file_csv = f"{AGENCY_ASSIGNS_DIR}/{summary_file_common}.csv"
        self.summary_file_jpg = f"{AGENCY_IMAGES_DIR}/{summary_file_common}.jpg"
        self.summary_file_name = self.summary_file_name.format(
            self.sample_inst_config.selected_agency, self.sample_inst_config.selected_date
        )
        self.summary_file_csv = self.summary_file_csv.format(
            self.sample_inst_config.selected_agency, self.sample_inst_config.selected_date
        )
        self.summary_file_jpg = self.summary_file_jpg.format(
            self.sample_inst_config.selected_agency, self.sample_inst_config.selected_date
        )

        self.gen_data_prefixes = (
            self.sample_inst_config.selected_agency,
            self.sample_inst_config.selected_date,
            self.sample_inst_config.no_of_trips,
            self.sample_inst_config.no_of_runs,
            self.sample_inst_config.no_of_vehicles,
            self.sample_inst_config.random_seed
        )
        data = GenDataLoader.instance().data
        self.source = Node(
            0, data['true_locations'][0], 0, DAY_IN_MINUTES * MIN_SEC_SWAP, "source", 0, 0
        )
        self.sink = Node(
            0, data['true_locations'][0], 0, DAY_IN_MINUTES * MIN_SEC_SWAP, "sink", 0, 0
        )
        self.cost = 0
        self.operational_cost = 0
        self.total_duration = 0
        self.total_travel_time = 0
        self.no_of_requests = 0
        self.no_of_runs = 0
        self.no_of_vehicles = 0
        self.run_plans = {}
        self.requests = []
        self._runs = []
        self._vehicles = []
        self._params = {}
        self._threshold_limit = -1
        self._assign_runs = False
        self._args = custom_args
        for i, node in enumerate(data['pick_up_nodes']):
            booking_id = data["booking_ids"][i]
            if booking_ids is not None and int(booking_id) not in booking_ids:
                continue
            p_idx_n = 2 * i + 1
            p_true_loc = data['true_locations'][p_idx_n]
            p_capacity = data['capacities'][i]
            if i in data['wheel_chair_counts']:
                pick_up_wheel_chair_count = data['wheel_chair_counts'][i]
            else:
                # fixed for exact_passenger_capacity = False
                pick_up_wheel_chair_count = 0
            pick_up_node = Node(
                p_idx_n,
                p_true_loc,
                data['pick_up_time_windows'][i][0],
                data['pick_up_time_windows'][i][1],
                "pick_up",
                pick_up_wheel_chair_count,
                p_capacity
            )
            d_idx_n = p_idx_n + 1
            d_true_loc = data['true_locations'][d_idx_n]
            d_capacity = -1 * p_capacity
            if i in data['wheel_chair_counts']:
                drop_off_wheel_chair_count_temp = data['wheel_chair_counts'][i]
            else:
                # fixed for exact_passenger_capacity = False
                drop_off_wheel_chair_count_temp = {}
            drop_off_wheel_chair_count = {}
            for _key in drop_off_wheel_chair_count_temp:
                drop_off_wheel_chair_count[_key] = -1 * drop_off_wheel_chair_count_temp[_key]

            drop_off_node = Node(
                d_idx_n,
                d_true_loc,
                data['drop_off_time_windows'][i][0],
                data['drop_off_time_windows'][i][1],
                "drop_off",
                drop_off_wheel_chair_count,
                d_capacity
            )
            req = Request(
                pick_up_node, drop_off_node,
                data["client_ids"][i], data["booking_ids"][i], data["start_times"][i],
                data["pick_up_broad_time_windows"][i], data["travel_times"][i],
                data["pick_up_zips"][i], data["drop_off_zips"][i]
            )
            self.requests.append(req)

        if custom_args is not None:
            str_params = ["consider_neg_wait"]
            float_params = [
                "wait_time", "assign_fraction_cost", "score_threshold",
                "assign_fraction_threshold", "length_of_run", "soft_max_param",
            ]
            for str_param in str_params:
                if str_param in custom_args.__dict__.keys():
                    self._params[str_param] = str(custom_args.__dict__[str_param])

            for float_param in float_params:
                if float_param in custom_args.__dict__.keys():
                    self._params[float_param] = float(custom_args.__dict__[float_param])

    def copy(self):
        """
        this function will create the copy of the original instance
        :return: the copy the original object
        """
        cs_cpy = type(self)(self.prefix, self.sample_inst_config, self.config, self._args)
        cs_cpy.source = self.source
        cs_cpy.sink = self.sink
        cs_cpy.cost = self.cost
        cs_cpy.total_duration = self.total_duration
        cs_cpy.total_travel_time = self.total_travel_time
        cs_cpy.no_of_runs = self.no_of_runs
        cs_cpy.no_of_requests = self.no_of_requests
        cs_cpy.no_of_vehicles = self.no_of_vehicles
        cs_cpy.run_plans = self.run_plans.copy()
        cs_cpy.requests = [_req.copy() for _req in self.requests]
        cs_cpy._runs = [_run.copy() for _run in self._runs]
        cs_cpy._vehicles = [_veh.copy() for _veh in self._vehicles]
        cs_cpy._params = self._params.copy()
        cs_cpy._threshold_limit = self._threshold_limit
        cs_cpy._assign_runs = self._assign_runs
        cs_cpy._args = self._args
        return cs_cpy

    def get_summary_suffix(self):
        """
            return summary suffix
        """
        return f"{self.prefix}_t{self.sample_inst_config.no_of_trips}_" + \
               f"r{self.sample_inst_config.no_of_runs}_" + \
               f"v{self.sample_inst_config.no_of_vehicles}_" + \
               f"{self.sample_inst_config.random_seed}"

    @staticmethod
    def specify_req_by_bw(request):
        """
        :param request: incoming request
        :return: request with pick-up and drop-off window adjusted
        """
        travel_time = request.travel_time
        request.pick_up_node.earliest_arrival = request.broad_time_windows[0]
        request.pick_up_node.latest_arrival = request.broad_time_windows[1]
        request.drop_off_node.earliest_arrival = request.broad_time_windows[0] + travel_time
        request.drop_off_node.latest_arrival = request.broad_time_windows[1] + travel_time
        return request

    def set_bw_at_init(self):
        """
            Set the broad pickup window at the start
        """
        requests = self.requests.copy()
        self.requests = []
        for request in requests:
            self.requests.append(self.specify_req_by_bw(request))

    def update_request(self, request, idx):
        """
        update the request with optimal pick-up time window
        :param request: request with optimal pick-up time window
        :param idx: index of the request
        """
        if len(self.requests) > idx:
            exp_idx = int(request.pick_up_node.idx / 2)
            # this is to make sure the request is assigned to correct location
            if exp_idx == idx:
                self.requests[idx] = request
            else:
                raise ValueError(f"Invalid request assignment, assigning request {exp_idx} to position {idx}")
        else:
            raise IndexError(f"Invalid request index {idx}")

    def reset(self):
        self.cost = 0
        self.operational_cost = 0
        self.total_duration = 0
        self.total_travel_time = 0
        self.no_of_requests = 0
        self.no_of_runs = 0
        self.no_of_vehicles = 0
        self.run_plans = {}
        self._runs = []
        self._vehicles = []

    def get_size(self):
        """
        :return: number of requests
        """
        return len(self.requests)

    def get_requests(self):
        """
        :return: number of requests
        """
        return self.requests

    def get_actual_size(self):
        """
        :return: number of requests
        """
        return self.get_size()

    def get_runs(self):
        """
        :return: set of runs
        """
        return self._runs

    def get_vehicles(self):
        """
        :return: set of vehicles
        """
        return self._vehicles

    def set_runs(self, runs):
        """
        set the runs to the solver
        """
        if not isinstance(runs, list):
            raise ValueError(f"expected type: list, actual type: {type(runs)}")
        self._runs = runs

    def add_run(self, run):
        """
        :param run: add the run to existing runs
        """
        self._runs.append(run)

    def set_vehicles(self, vehicles):
        """
        set the vehicles to the solver
        """
        if not isinstance(vehicles, list):
            raise ValueError(f"expected type: list, actual type: {type(vehicles)}")
        self._vehicles = vehicles

    def modify_params(self, params):
        """
        :param params: set the parameters
        """
        self._params.update(params.copy())

    def set_threshold_limit(self, threshold_limit):
        """
        :param threshold_limit: set duration threshold limit
        """
        self._threshold_limit = threshold_limit

    def load_generic_data(self, dump_file_name=GENERIC_DATA_FILE_NAME):
        """
        :param dump_file_name: file name for loading the generic data for the solver
        :return: returns the travel time dictionary with respect to locations
        """
        file_name = dump_file_name.format(*self.gen_data_prefixes)
        if not dump_exists(file_name):
            data_obj = CSData(
                sample_inst_config=self.sample_inst_config,
                config=self.config.data_config
            )
        else:
            data_obj = load_obj(file_name)
        return data_obj

    def assign_requests(self, idx=-1):
        raise NotImplementedError

    def assign_requests_parallel(self, idx=-1):
        raise NotImplementedError

    def assign_runs(self, verify=False, keep_run_name=False):
        """
            assign the active runs to the vehicles
        """
        if not self._assign_runs:
            # bypass the assigning the runs
            return True
        if verify:
            req_count = 0
            for run in self._runs:
                req_count += run.get_size()
            if req_count != self.get_size():
                raise ImplementationError("Not all requests are assigned")
        self._vehicles = []
        assigned_runs = []
        active_runs = self.get_active_runs().copy()
        sorted_active_runs = sorted(active_runs, key=lambda x: x.get_start())
        success = False
        cur_idx = 0
        run_idx = 0
        while len(assigned_runs) < len(sorted_active_runs):
            cur_veh = Vehicle(f"veh_{cur_idx}")
            self._vehicles.append(cur_veh)
            while len(assigned_runs) < len(sorted_active_runs):
                success = False
                for i, run in enumerate(sorted_active_runs):
                    if run not in assigned_runs:
                        if cur_veh.check_assign(run):
                            success = cur_veh.assign(run)
                            if success:
                                if not keep_run_name:
                                    run.set_name(f"run_{run_idx}")
                                assigned_runs.append(run)
                                run_idx += 1
                                break
                if not success:
                    break
            cur_idx += 1
            if self.sample_inst_config.no_of_vehicles != -1:
                if cur_idx >= self.sample_inst_config.no_of_vehicles:
                    logger.error("Not enough vehicles !!!")
                    sys.exit(-1)
        return success

    def solve(self, write_summary=False):
        """
            assign the requests to runs, and runs to vehicles
        """
        detail = None
        if self._args is not None:
            agency = self._args.agency
            date = self._args.date
            detail = f"(agency :{agency}, date: {date})"
        self.reset()
        if self.assign_requests():
            if self.assign_runs():
                self.verify()
                if write_summary:
                    self.write_solution()
                    self.write_summary()
            else:
                raise ValueError(f"Unable to assign all the runs to vehicles {detail} !!!")
        else:
            raise ValueError(f"Unable to assign all the requests {detail} !!!")

    def solve_parallel(self, write_summary=False):
        """
            this is to speed up assigning process.
        """
        detail = None
        if self._args is not None:
            agency = self._args.agency
            date = self._args.date
            detail = f"(agency :{agency}, date: {date})"
        self.reset()
        if self.assign_requests_parallel():
            if self.assign_runs():
                self.verify()
                if write_summary:
                    self.write_solution()
                    self.write_summary()
            else:
                raise ValueError(f"Unable to assign all the runs to vehicles {detail} !!!")
        else:
            raise ValueError(f"Unable to assign all the requests {detail} !!!")

    def solve_limited(self, idx=0, write_summary=False):
        """
            assign the requests to runs, and runs to vehicles,
            but assign only first request up to the number specified by idx
        """
        detail = None
        if self._args is not None:
            agency = self._args.agency
            date = self._args.date
            detail = f"(agency :{agency}, date: {date})"
        self.reset()
        if self.assign_requests(idx):
            if self.assign_runs():
                if write_summary:
                    self.write_solution()
                    self.write_summary()
            else:
                raise ValueError(f"Unable to assign all the runs to vehicles {detail} !!!")
        else:
            raise ValueError(f"Unable to assign all the requests {detail} !!!")

    def get_params(self):
        """
        :return: parameter configuration
        """
        return self._params

    def get(self, factor):
        """
        :param factor: name of the statistics or details
        :return: get particular statistics or details
        """
        result = None
        self.compute_cost()
        if factor == "cost":
            result = self.cost
        elif factor == "run":
            result = self.no_of_runs
        elif factor == "vehicle":
            result = self.no_of_vehicles
        return result

    def compute_cost(self):
        """
        compute the cost, based on the runs assignments and vehicle assignments
        :return: return the combined objective cost
        """
        self.cost = 0
        self.operational_cost = 0
        self.total_duration = 0
        self.total_travel_time = 0
        self.no_of_runs = 0
        self.no_of_vehicles = 0
        self.no_of_requests = 0
        for run in self._runs:
            if run.is_active():
                self.no_of_runs += 1
                self.total_duration += run.get_duration()
                self.total_travel_time += run.get_total_travel_time()
                self.no_of_requests += run.get_size()
        self.operational_cost = self.cost
        for vehicle in self._vehicles:
            if vehicle.is_assigned():
                self.no_of_vehicles += 1
        run_weight = GenDataLoader.instance().agency_config.run_weight
        self.cost = self.total_duration + run_weight * self.no_of_runs
        return self.cost

    def get_active_runs(self):
        """
        :return: the list of active runs (in baseline words the runs serves at-least one requests)
        """
        active_runs = []
        for run in self._runs:
            if run.is_active():
                active_runs.append(run)
        return active_runs

    def get_overlapping_runs(self):
        """
        filter out the runs to have overlapping operations
        to make it merge or split and merge
        :return: return the overlapping runs
        """
        active_runs = self.get_active_runs().copy()
        random.shuffle(active_runs)
        filtered_runs = []
        first_run = None
        ignored_runs = []
        for run in active_runs:
            if first_run is None:
                first_run = run
            if first_run.is_overlap(run):
                filtered_runs.append(run)
            else:
                ignored_runs.append(run)

        # if there is no overlapping runs found try to add some random run
        if len(filtered_runs) == 1:
            if len(ignored_runs) > 0:
                filtered_runs += [random.choice(ignored_runs)]
        return filtered_runs

    def swap(self, verify=False):
        """
        do the swap operation
        :return: return the swapped solution
        """
        filtered_runs = self.get_overlapping_runs()
        if len(filtered_runs) > 2:
            chosen_runs = random.sample(filtered_runs, 2)
            run_1 = chosen_runs[0].copy()
            run_2 = chosen_runs[1].copy()
            request_1 = random.choice(run_1.get_requests())
            request_2 = random.choice(run_2.get_requests())
            run_1.remove_request(request_1)
            run_2.remove_request(request_2)
            status_1 = run_1.check_and_insert(request=request_2, is_from_mutation=True)
            status_2 = run_2.check_and_insert(request=request_1, is_from_mutation=True)
            max_duration = GenDataLoader.instance().agency_config.run_max_duration
            if self._threshold_limit > 0:
                max_duration = self._threshold_limit
            if status_1 and status_2 and run_1.get_duration() <= max_duration and run_2.get_duration() <= max_duration:
                idx_1 = self._runs.index(chosen_runs[0])
                idx_2 = self._runs.index(chosen_runs[1])
                self._runs[idx_1] = run_1
                self._runs[idx_2] = run_2
                self.assign_runs(verify=verify)
                if verify:
                    self.verify()
                logger.info("alteration swap success")
        return self

    def permutation(self, n=2, verify=False):
        """
        do the swap operation in a single run
        :return: return the swapped solution
        """
        filtered_runs = self.get_overlapping_runs()
        if len(filtered_runs) > 2:
            chosen_run = random.choice(filtered_runs)
            run = chosen_run.copy()
            failed, updated_run = run.permutation(n, self._threshold_limit)
            if not failed:
                idx = self._runs.index(chosen_run)
                self._runs[idx] = updated_run
                self.assign_runs(verify=verify)
                if verify:
                    self.verify()
                logger.info("permutation success")
        return self

    def split_and_merge_runs(self, n=2, fixed_pattern=True, verify=False):
        """
        do the merge operation
        :param n: number of runs
        :param fixed_pattern: this will make a fix pattern
        :param verify: enabling variable will verify the implementation
        :return: return the merged solution
        """
        filtered_runs = self.get_overlapping_runs()
        if len(filtered_runs) > n:
            chosen_runs = random.sample(filtered_runs, n)
            merged_runs, is_merged = chosen_runs[0].split_and_merge(
                chosen_runs[1:], fixed_pattern=fixed_pattern, threshold_limit=self._threshold_limit
            )
            max_duration = GenDataLoader.instance().agency_config.run_max_duration
            if self._threshold_limit > 0:
                max_duration = self._threshold_limit
            status = []
            for run in merged_runs:
                status.append(run.get_duration() <= max_duration)
            if is_merged and all(status):
                for chosen_run in chosen_runs:
                    self._runs.remove(chosen_run)
                for merged_run in merged_runs:
                    self._runs.append(merged_run)
                self.assign_runs(verify=verify)
                if verify:
                    self.verify()
                logger.info("alteration split and merge success")
        return self

    def verify(self):
        """
        This function will verify whether there is any error in the implementation
        :return: status of verification
        """
        expected_req = len(self.get_requests())
        checked_nodes = []
        for request in self.get_requests():
            for run in self._runs:
                if run.is_active():
                    if run.verify():
                        c_p = run.get_full_indices().contains(request.pick_up_node.idx) and \
                              request.pick_up_node.idx not in checked_nodes
                        c_d = run.get_full_indices().contains(request.drop_off_node.idx) and \
                              request.drop_off_node.idx not in checked_nodes
                        if c_p and c_d:
                            checked_nodes.append(request.pick_up_node.idx)
                            checked_nodes.append(request.drop_off_node.idx)
                            expected_req -= 1
                            break
                        else:
                            if request.pick_up_node.idx in checked_nodes:
                                raise ImplementationError(
                                    f"Error in implementation, double insert {request.pick_up_node.idx} !!!"
                                )
                            if request.drop_off_node.idx in checked_nodes:
                                raise ImplementationError(
                                    f"Error in implementation, double insert {request.drop_off_node.idx} !!!"
                                )

                    else:
                        raise ImplementationError("Error in implementation, in-side run !!!")

        if expected_req:
            raise ImplementationError(f"Error in implementation, in assigning req !!!")
        
        if self._assign_runs:
            expected_run = len(self.get_active_runs())
            checked_runs = []
            for run in self._runs:
                if run.is_active():
                    for veh in self._vehicles:
                        if veh.is_assigned():
                            if run in veh.get_runs() and run not in checked_runs:
                                checked_runs.append(run)
                                expected_run -= 1
                                break
                            elif run in checked_runs:
                                raise ImplementationError(f"Error in implementation, double insert {run.name} !!!")
            if expected_run:
                raise ImplementationError("Error in implementation, in assigning run !!!")

    def write_summary(self):
        """
            write the summary of the solution
        """
        create_dir(self.summary_file_name)
        summary_file = open(self.summary_file_name, "w+")
        summary_file.write("[RUN STATISTICS]\n")
        for run in self._runs:
            if run.is_active():
                summary_file.write(f"{run.get_route()}\n")
        for vehicle in self._vehicles:
            if vehicle.is_assigned():
                summary_file.write(f"{vehicle.get_assign()}\n")
        self.compute_cost()
        summary_file.write("[OVERALL STATISTICS]\n")
        summary_file.write(f"Number of requests: {self.no_of_requests}\n")
        summary_file.write(f"Objective of the problem: {self.cost}\n")
        summary_file.write(f"Number of runs used: {self.no_of_runs}\n")
        if self._assign_runs:
            summary_file.write(f"Number of vehicles used: {self.no_of_vehicles}\n")
        summary_file.write(f"Total travel duration: {convert_sec_to_hh_mm_ss(self.total_duration)}\n")
        summary_file.close()

    def write_solution(self):
        """
            write solution to file
        """
        import pandas as pd
        df_assign = pd.read_csv(SHORTEN_DATA_FILE_NAME.format(*self.gen_data_prefixes))
        trip_assignments = [None] * len(df_assign)
        vehicle_assignments = [None] * len(df_assign)
        vehicle_run_dict = {}
        if self._assign_runs:
            for vehicle in self._vehicles:
                if vehicle.is_assigned():
                    for run in vehicle.get_runs():
                        vehicle_run_dict[run.get_name()] = vehicle.get_name()
        for run in self._runs:
            if run.is_active():
                for node in run.get_nodes():
                    if node.idx % 2 == 1:
                        pos = int(node.idx/2)
                        trip_assignments[pos] = run.get_name()
                        if self._assign_runs:
                            vehicle_assignments[pos] = vehicle_run_dict[run.get_name()]
        df_assign["Run Assignments"] = trip_assignments
        if len(trip_assignments) == len(vehicle_assignments):
            if self._assign_runs:
                df_assign["Vehicle Assignments"] = vehicle_assignments
        df_assign = df_assign.dropna(subset=['Run Assignments'])
        create_dir(self.summary_file_csv)
        df_assign.to_csv(self.summary_file_csv, index=False)

    def visualize(self):
        """
            this function will visualize the assignment
        """
        for run in self._runs:
            if run.is_active():
                self.run_plans[run.get_name()] = run.get_assign_data()
        visualize(
            self,
            GenDataLoader.instance().agency_config.depot_coordinates,
            GenDataLoader.instance().agency_config.area_box_limits
        )

    def get_solution_details(self):
        if self._assign_runs:
            return f"{self.get_size()},{self.no_of_runs},{self.no_of_vehicles}," + \
                   f"{self.cost},{self.operational_cost},{self.total_duration},{self.total_travel_time}"
        else:
            return f"{self.get_size()},{self.no_of_runs},," + \
                   f"{self.cost},{self.operational_cost},{self.total_duration},{self.total_travel_time}"
