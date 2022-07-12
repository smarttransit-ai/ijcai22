##################################################################
#                 VROOM WRAPPER IMPLEMENTATION                   #
##################################################################
#  In this wrapper implementation, I assume that the vehicles    #
#  can start around every 1/2 hours and can serve upto 10 hours  #
#  (i.e., the maximum duration of the runs) after generating the #
#  runs, I assign the runs to vehicles                           #
##################################################################

import json
import os

from base.data.GenDataLoader import GenDataLoader
from base.model.BaseLineWrapper import BaseLineWrapper
from common.constant.constants import AGENCY_PLANS_DIR, DAY_IN_MINUTES, MIN_SEC_SWAP
from common.util.common_util import create_dir


class VRoomBasedPTOpt(BaseLineWrapper):
    """
        This is a wrapper class for VROOM VRP solver
    """

    def __init__(self, run_cls, sample_inst_config=None, config=None, custom_args=None):
        super(VRoomBasedPTOpt, self).__init__("vroom", run_cls, sample_inst_config, config, custom_args)
        summary_file_common = self.get_summary_suffix()
        self.vroom_input_json = f"{AGENCY_PLANS_DIR}/{summary_file_common}.json"
        self.vroom_solution_json = f"{AGENCY_PLANS_DIR}/{summary_file_common}_sol.json"
        self.vroom_input_json = self.vroom_input_json.format(
            self.sample_inst_config.selected_agency, self.sample_inst_config.selected_date
        )
        self.vroom_solution_json = self.vroom_solution_json.format(
            self.sample_inst_config.selected_agency, self.sample_inst_config.selected_date
        )
        self.trash_intermediate_files = True

    def copy(self):
        """
        this function will create the copy of the original instance
        :return: the copy the original object
        """
        vroom_cpy = type(self)(self._run_cls, self.sample_inst_config, self.config, self._args)
        vroom_cpy.source = self.source
        vroom_cpy.sink = self.sink
        vroom_cpy.cost = self.cost
        vroom_cpy.total_duration = self.total_duration
        vroom_cpy.total_travel_time = self.total_travel_time
        vroom_cpy.no_of_runs = self.no_of_runs
        vroom_cpy.no_of_requests = self.no_of_requests
        vroom_cpy.no_of_vehicles = self.no_of_vehicles
        vroom_cpy.run_plans = self.run_plans.copy()
        vroom_cpy.requests = [_req.copy() for _req in self.requests]
        vroom_cpy._run_cls = self._run_cls
        vroom_cpy._runs = [_run.copy() for _run in self._runs]
        vroom_cpy._vehicles = [_veh.copy() for _veh in self._vehicles]
        vroom_cpy._params = self._params.copy()
        vroom_cpy._threshold_limit = self._threshold_limit
        vroom_cpy._assign_runs = self._assign_runs
        vroom_cpy._args = self._args
        return vroom_cpy

    def assign_requests(self, idx=-1):
        sel_requests = self.get_requests().copy()
        if idx != -1:
            sel_requests = sel_requests[:idx + 1]
        no_of_requests = len(sel_requests)

        def create_vehicle():
            """
                :return: list of vehicle JSON objects
            """
            _vehicles = []
            if GenDataLoader.instance().agency_config.run_max_duration < DAY_IN_MINUTES * MIN_SEC_SWAP:
                # when there is a limitation in the run-duration
                for _idx in range(no_of_requests):
                    # technically the range should be 0-28, but the request may go to the
                    # next day, so providing few more hour ranges to accommodate those requests
                    for _k in range(0, 35):
                        _vehicle = {
                            "id": _idx + _k * 35,
                            "start_index": 0,
                            "end_index": 0,
                            "capacity": [GenDataLoader.instance().agency_config.vehicle_capacity],
                            "skills": [1],
                            "time_window": [_k * 1800, (_k + 20) * 1800]
                        }
                        _vehicles.append(_vehicle)
            else:
                for _idx in range(no_of_requests):
                    _vehicle = {
                        "id": _idx,
                        "start_index": 0,
                        "end_index": 0,
                        "capacity": [GenDataLoader.instance().agency_config.vehicle_capacity],
                        "skills": [1],
                    }
                    _vehicles.append(_vehicle)
            return _vehicles

        def create_shipment():
            """
                :return: list of requests, as shipments
            """
            _shipments = []
            _nodes = {}
            _size = len(sel_requests)
            for _request in sel_requests:
                _shipment = {
                    "amount": [int(_request.pick_up_node.capacity)],
                    "skills": [1],
                    "pickup": {
                        "id": _request.pick_up_node.idx,
                        "service": 0,
                        "location_index": _request.pick_up_node.idx,
                        "time_windows": [
                            [
                                int(_request.pick_up_node.earliest_arrival),
                                int(_request.pick_up_node.latest_arrival)
                            ]
                        ]
                    },
                    "delivery": {
                        "id": _request.drop_off_node.idx,
                        "service": 0,
                        "location_index": _request.drop_off_node.idx,
                        "time_windows": [
                            [
                                int(_request.drop_off_node.earliest_arrival),
                                int(_request.drop_off_node.latest_arrival)
                            ]
                        ]
                    }
                }
                _shipments.append(_shipment)

            return _shipments

        def create_time_matrix():
            """
                :return: the travel-time matrix
            """
            data = GenDataLoader.instance().data
            _matrix_entries = []
            _true_locations = data["true_locations"]
            _dwell_time = GenDataLoader.instance().agency_config.dwell_time
            for _i in sorted(_true_locations.keys()):
                _row_times = []
                for _j in sorted(_true_locations.keys()):
                    _time = 0
                    _loc_i = _true_locations[_i]
                    _loc_j = _true_locations[_j]
                    if _loc_i != _loc_j:
                        _, _time = GenDataLoader.instance().get_travel_time(_loc_i, _loc_j)
                    _row_times.append(_time + _dwell_time)
                _matrix_entries.append(_row_times)
            return _matrix_entries

        vehicles = create_vehicle()
        shipments = create_shipment()
        matrix = create_time_matrix()

        input_data = {
            "vehicles": vehicles,
            "shipments": shipments,
            "matrices": {
                "car": {
                    "durations": matrix
                }
            },
        }

        input_file = self.vroom_input_json.replace(".json", f"{idx}.json")
        solution_file = self.vroom_solution_json.replace(".json", f"{idx}.json")
        create_dir(input_file)

        # save the input data for the vroom
        with open(input_file, "w+") as json_file:
            json.dump(input_data, json_file, indent=2)

        # run the vroom command and store the output
        # in the specified output file
        from sys import platform
        if platform == "linux" or platform == "linux2":
            vroom_location = "common/vroom_linux"
        elif platform == "darwin":
            vroom_location = "common/vroom_macosx"
        else:
            vroom_location = "common/vroom_linux"
        command = f"{vroom_location} -i {input_file} -o {solution_file} -t {int(self._args.no_of_workers)} -x 5"
        os.system(command)

        all_requests = []
        with open(solution_file, "r+") as json_file:
            output_data = json.load(json_file)
            unassigned = output_data['summary']['unassigned']

            # if all requests are assigned
            if unassigned == 0:
                # extract the assignment from the output and compute the results
                # to match with our heuristics
                nodes = {}
                pick_up_request = {}
                for request in sel_requests:
                    nodes[request.pick_up_node.idx] = request.pick_up_node
                    nodes[request.drop_off_node.idx] = request.drop_off_node
                    pick_up_request[request.pick_up_node.idx] = request

                for cur_idx, route in enumerate(output_data['routes']):
                    requests = []
                    time_values = []
                    route_nodes = []
                    for step in route['steps']:
                        if 'id' in step:
                            selected_node_idx = step['id']
                            wait_time = step['waiting_time']
                            time_values.append(step['arrival'] + wait_time)
                            route_nodes.append(nodes[selected_node_idx])
                            if selected_node_idx in pick_up_request.keys():
                                requests.append(pick_up_request[selected_node_idx])
                    total_travel_time = route['cost']
                    if len(requests) > 0 and len(route_nodes) > 0:
                        cur_run = self._run_cls(f"run_{cur_idx}", self.source, self.sink, self._params)
                        success = cur_run.check_and_force_insert_bw(
                            requests=requests, time_values=time_values, temp_nodes=route_nodes,
                            total_travel_time=total_travel_time
                        )
                        if success:
                            all_requests.extend(requests)
                            self._runs.append(cur_run)

            # saving the formatted results file
            with open(solution_file, "w+") as json_formatted_file:
                json.dump(output_data, json_formatted_file, indent=2)

        if self.trash_intermediate_files:
            os.remove(solution_file)
            os.remove(input_file)
        return len(all_requests) == len(sel_requests)

    def get_solution_details(self):
        return f"{self.get_size()},{self.no_of_runs},{self.no_of_vehicles}," + \
               f"{self.cost},{self.operational_cost},{self.total_travel_time}"
