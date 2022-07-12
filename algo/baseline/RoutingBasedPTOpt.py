##################################################################
#              GOOGLE OR-TOOL ROUTING IMPLEMENTATION             #
##################################################################
#  In this wrapper implementation for Google OR tool routing     #
##################################################################

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from base.data.GenDataLoader import GenDataLoader
from base.model.BaseLineWrapper import BaseLineWrapper
from common.constant.constants import VEHICLE_TRAVEL_MAX_TIME, DAY_IN_MINUTES, MIN_SEC_SWAP
from common.util.common_util import logger


class RoutingBasedPTOpt(BaseLineWrapper):
    """
        This is a wrapper class for Google OR-Tools VRP solver
    """

    def __init__(self, run_cls, sample_inst_config=None, config=None, custom_args=None):
        super(RoutingBasedPTOpt, self).__init__("routing", run_cls, sample_inst_config, config, custom_args)
        self.__skip_first_sol_strategy = False
        if custom_args is not None:
            self.__search_duration = int(custom_args.plan_duration)
        else:
            self.__search_duration = 300
        self.__first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        self.__meta_heuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC

    def copy(self):
        """
        this function will create the copy of the original instance
        :return: the copy the original object
        """
        rou_cpy = type(self)(self._run_cls, self.sample_inst_config, self.config, self._args)
        rou_cpy.source = self.source
        rou_cpy.sink = self.sink
        rou_cpy.cost = self.cost
        rou_cpy.total_duration = self.total_duration
        rou_cpy.total_travel_time = self.total_travel_time
        rou_cpy.no_of_runs = self.no_of_runs
        rou_cpy.no_of_requests = self.no_of_requests
        rou_cpy.no_of_vehicles = self.no_of_vehicles
        rou_cpy.run_plans = self.run_plans.copy()
        rou_cpy.requests = [_req.copy() for _req in self.requests]
        rou_cpy._run_cls = self._run_cls
        rou_cpy._runs = [_run.copy() for _run in self._runs]
        rou_cpy._vehicles = [_veh.copy() for _veh in self._vehicles]
        rou_cpy._params = self._params.copy()
        rou_cpy._threshold_limit = self._threshold_limit
        rou_cpy._assign_runs = self._assign_runs
        rou_cpy._args = self._args
        return rou_cpy

    def set_configs(
            self,
            search_duration,
            first_sol_strategy=routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
            meta_heuristic=routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
            skip_first_sol_strategy=False
    ):
        """
        :param search_duration: define the maximum searching duration
        :param first_sol_strategy: first solution strategy
        :param meta_heuristic: meta heuristic to search the solution
        :param skip_first_sol_strategy: enabling this will skip generating initial solution using custom
        algorithm
        """
        self.__search_duration = search_duration
        self.__first_solution_strategy = first_sol_strategy
        self.__meta_heuristic = meta_heuristic
        self.__skip_first_sol_strategy = skip_first_sol_strategy

    def assign_requests(self, idx=-1):
        sel_requests = self.get_requests().copy()
        if idx != -1:
            sel_requests = sel_requests[:idx + 1]

        data = GenDataLoader.instance().data
        time_matrix = GenDataLoader.instance().gen_data.get_time_matrix()
        number_of_vehicles = len(sel_requests)

        # create the routing manager
        manager = pywrapcp.RoutingIndexManager(
            2 * len(sel_requests) + 1, number_of_vehicles, 0
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def setup_general_constraints():
            for i, _request in enumerate(sel_requests):
                pickup_index = manager.NodeToIndex(_request.pick_up_node.idx)
                delivery_index = manager.NodeToIndex(_request.drop_off_node.idx)
                routing.AddPickupAndDelivery(pickup_index, delivery_index)
                # this will ensure the pick-up and drop-off happen using same vehicle.
                routing.solver().Add(
                    routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
                )

        def setup_time_constraints():
            dwell_time = GenDataLoader.instance().agency_config.dwell_time

            # Create and register a transit callback.
            def time_callback(from_index, to_index):
                # Convert from routing variable Index to time matrix NodeIndex.
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                # this fix is required to perform the dwell time
                if from_node == to_node:
                    time_val = 0
                else:
                    time_val = time_matrix[from_node][to_node]
                    time_val += dwell_time
                return time_val

            transit_callback_index = routing.RegisterTransitCallback(time_callback)

            # define the time cost of each args
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            routing.SetFixedCostOfAllVehicles(GenDataLoader.instance().agency_config.run_weight)

            # Add Time Windows constraint.
            time = 'Time'
            routing.AddDimension(
                transit_callback_index,
                GenDataLoader.instance().agency_config.wait_time,  # maximum wait time
                VEHICLE_TRAVEL_MAX_TIME,  # maximum time a vehicle can reach
                False,  # Don't force start cumulative to zero.
                time)

            time_dimension = routing.GetDimensionOrDie(time)

            # Add time window constraints for each location except depot.
            time_windows = [[0, 2 * DAY_IN_MINUTES * MIN_SEC_SWAP]]
            for _req in sel_requests:
                time_windows.append([_req.pick_up_node.earliest_arrival, _req.pick_up_node.latest_arrival])
                time_windows.append([_req.drop_off_node.earliest_arrival, _req.drop_off_node.latest_arrival])

            for location_idx, time_window in enumerate(time_windows):
                if location_idx == 0:
                    continue
                _index = manager.NodeToIndex(location_idx)
                time_dimension.CumulVar(_index).SetRange(time_window[0], time_window[1])
                if location_idx != 0 and location_idx % 2 == 0:
                    _p_index = manager.NodeToIndex(location_idx - 1)
                    max_detour_time = GenDataLoader.instance().agency_config.max_detour_time
                    travel_and_load = time_matrix[_p_index][_p_index]
                    upper_limit = max_detour_time + travel_and_load
                    routing.solver().Add(
                        time_dimension.CumulVar(_index) - time_dimension.CumulVar(_p_index) <= upper_limit
                    )

            # Add time window constraints for each run start node.
            for _veh_id in range(number_of_vehicles):
                _idx = routing.Start(_veh_id)
                time_dimension.CumulVar(_idx).SetRange(time_windows[0][0], time_windows[0][1])

            # Instantiate route start and end times to produce feasible times.
            upper_limit = GenDataLoader.instance().agency_config.run_max_duration
            for _veh_id in range(number_of_vehicles):
                time_dimension.SetSpanCostCoefficientForVehicle(1, _veh_id)
                time_dimension.SetSpanUpperBoundForVehicle(upper_limit, _veh_id)

        def setup_occupancy_constraints():
            capacities = [0]
            for i, capacity in enumerate(data['capacities']):
                capacities.append(capacity)
                capacities.append(0)

            # Add Capacity constraint.
            def demand_callback(from_index):
                # Convert from routing variable Index to demands NodeIndex.
                from_node = manager.IndexToNode(from_index)
                return capacities[from_node]

            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

            vehicle_capacity = GenDataLoader.instance().agency_config.vehicle_capacity

            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # null capacity slack
                [vehicle_capacity for _ in range(number_of_vehicles)],  # vehicle maximum capacities
                True,  # start cumulative to zero
                'Capacity')

        setup_general_constraints()
        setup_time_constraints()
        setup_occupancy_constraints()

        # meta heuristics
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        if not self.__skip_first_sol_strategy:
            search_parameters.first_solution_strategy = self.__first_solution_strategy
        search_parameters.local_search_metaheuristic = self.__meta_heuristic
        search_parameters.log_search = True
        search_parameters.lns_time_limit.FromSeconds(self.__search_duration)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        all_requests = []
        if solution:
            # extract the assignment from the output and compute the results
            # to match with our heuristics

            nodes = {}
            pick_up_request = {}
            for request in sel_requests:
                nodes[request.pick_up_node.idx] = request.pick_up_node
                nodes[request.drop_off_node.idx] = request.drop_off_node
                pick_up_request[request.pick_up_node.idx] = request

            all_requests = []
            time_dimen = routing.GetDimensionOrDie("Time")
            for veh_id in range(number_of_vehicles):
                requests = []
                time_values = []
                route_nodes = []
                index = routing.Start(veh_id)
                total_travel_time = 0
                while not routing.IsEnd(index):
                    index = solution.Value(routing.NextVar(index))
                    first_location = None
                    if index in nodes.keys():
                        first_location = nodes[index].real_location
                    time_values.append(solution.Value(time_dimen.CumulVar(index)))
                    if index in nodes.keys() and nodes[index] not in route_nodes:
                        route_nodes.append(nodes[index])
                        next_idx = solution.Value(routing.NextVar(index))
                        if next_idx in nodes.keys() and first_location is not None:
                            second_location = nodes[next_idx].real_location
                            _, travel_time = GenDataLoader.instance().get_travel_time(first_location, second_location)
                            total_travel_time += travel_time
                    if index in pick_up_request.keys() and pick_up_request[index] not in requests:
                        requests.append(pick_up_request[index])

                if len(requests) > 0 and len(route_nodes) > 0:
                    cur_run = self._run_cls(f"run_{veh_id}", self.source, self.sink, self._params)
                    success = cur_run.check_and_force_insert_bw(
                        requests=requests, time_values=time_values[:-1], temp_nodes=route_nodes,
                        total_travel_time=total_travel_time
                    )
                    if success:
                        all_requests.extend(requests)
                        self._runs.append(cur_run)
        else:
            # copied error codes from
            # https://developers.google.com/optimization/routing/routing_options#search-status for provide better
            # description
            error_codes = {
                0: "Problem not solved yet.",
                1: "Problem solved successfully.",
                2: "No Solution found to the problem. Check the implementation for errors",
                3: "Time limit reached before finding a solution."
                   f"current time limit {self.__search_duration} seconds, increase the search duration and try",
                4: "Model, model parameters, or flags are not valid. Check the combinations of FS and MH"
            }
            error_message = f"Failure, reason: {error_codes[routing.status()]}"
            logger.error(error_message)
        return len(sel_requests) == len(all_requests)

    def get_solution_details(self):
        return f"{self.get_size()},{self.no_of_runs},{self.no_of_vehicles}," + \
               f"{self.cost},{self.operational_cost},{self.total_travel_time}"
