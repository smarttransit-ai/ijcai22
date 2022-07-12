##########################################
#       ABSTRACT RUN IMPLEMENTATION      #
##########################################

from collections import namedtuple
from enum import Enum

from base.common.DoubleMap import RequestDoubleMap
from base.common.MappedList import MappedList
from base.data.GenDataLoader import GenDataLoader
from base.entity.Node import NodeType
from base.errors import ImplementationError
from common.util.common_util import convert_sec_to_hh_mm_ss, logger

GenericAssignData = namedtuple("GenericAssignData", ["s_cor", "s_e", "s_l", "s_idx",
                                                     "e_cor", "e_e", "e_l", "e_idx", "tt"])

InsertStat = namedtuple(
    'InsertStat',
    ['nodes', 'times', 'node_times', 'capacities', 'src_time', 'snk_time', 'travel_time', 'distance', 'extra_time',
     'wait_time', 'busyness_pdh', 'busyness_ph', 'busyness_dh', 'busyness_h', 'tightness', 'extra_distance']
)


class RunErrorTypes(Enum):
    TimeViolation = 1
    CapacityViolation = 2
    TravelTimeViolation = 3
    RunDurationViolation = 4


# object that store adjusted times
# distances also computed along side
AdjustedTimes = namedtuple(
    "AdjustedTimes",
    ["success", "times", "node_times", "distances", 'src_time', 'snk_time', "duration", "travel_time", "distance"]
)

# object that stores adjusted capacities
AdjustedCapacities = namedtuple(
    "AdjustedCapacities",
    ["success", "capacities"]
)

# object to store the adjusted times and capacities
AdjustedValues = namedtuple(
    "AdjustedValues",
    ["success", "time_obj", "capacity_obj"]
)

# object to store the time-window violations
TimeWindowViolationStat = namedtuple(
    "TimeWindowViolationStat",
    ["request", "travel_time", "dwell_time", "actual_gap", "prev_req", "next_req", "prev_node", "next_node", "try_idx"]
)

# object to store the capacity violations
CapacityViolationStat = namedtuple(
    "CapacityViolationStat",
    ["request", "prev_req", "next_req", "prev_node", "next_node", "try_idx"]
)

# object to store run length violations
RunViolationStat = namedtuple(
    "RunViolationStat", ["request", "try_idx"]
)

# object to store the request insertion violations (in overall)
ReqInsertViolationStat = namedtuple(
    "ReqInsertViolationStat", ["request", "p_indices", "d_indices", "try_idx"]
)


class AbstractRun(object):
    """
        This class stores the details of Run.
        Run represents the ``route'' of each vehicle
    """

    def __init__(self, name, source, sink, params):
        self._name = name
        self._max_duration = GenDataLoader.instance().agency_config.run_max_duration
        self._source = source
        self._sink = sink
        self._params = params
        self._start = -1
        self._end = -1
        self._travel_time = 0
        self._distance = -1
        self._nodes = []
        self._requests = []
        self._times = []
        self._node_times = {}
        self._capacities = []
        self._log_error = False
        self._error_log = []
        self._error_hits = {}
        self._error_types_stats = {}
        self._time_window_violations = {}
        self._insertion_violations = {}
        self._capacity_violations = {}
        self._run_violations = {}
        self._cur_request = None
        self._try_index = 0
        self._pick_drop_off_map = RequestDoubleMap()

    def copy(self):
        """
        :return: copy the run object
        """
        run_copy = type(self)(self._name, self._source, self._sink, self._params)
        run_copy._start = self._start
        run_copy._end = self._end
        run_copy._travel_time = self._travel_time
        run_copy._distance = self._distance
        run_copy._nodes = self._nodes.copy()
        run_copy._requests = self._requests.copy()
        run_copy._times = self._times.copy()
        run_copy._node_times = self._node_times.copy()
        run_copy._capacities = self._capacities.copy()
        run_copy._log_error = self._log_error
        run_copy._error_log = self._error_log.copy()
        run_copy._error_hits = self._error_hits.copy()
        run_copy._error_types_stats = self._error_types_stats.copy()
        run_copy._time_window_violations = self._time_window_violations.copy()
        run_copy._insertion_violations = self._insertion_violations.copy()
        run_copy._capacity_violations = self._capacity_violations.copy()
        run_copy._run_violations = self._run_violations.copy()
        run_copy._cur_request = self._cur_request
        run_copy._try_index = self._try_index
        run_copy._pick_drop_off_map = self._pick_drop_off_map.copy()
        return run_copy

    def get_name(self):
        """
        :return: the name of the run
        """
        return self._name

    def set_name(self, name):
        """
        :param name: custom name
        """
        self._name = name

    def get_nodes(self):
        """
        :return: request nodes (both pick-up and drop-off nodes)
        """
        return self._nodes

    def get_size(self):
        """
        :return: the number of requests in the run
        """
        return len(self._requests)

    def get_requests(self):
        """
        :return: requests
        """
        return self._requests

    def get_node_indices(self):
        """
        :return: node indices
        """
        return MappedList([request.pick_up_node.idx for request in self._requests])

    def get_full_indices(self):
        """
        :return: full node indices
        """
        return MappedList([node.idx for node in self._nodes])

    def is_active(self):
        """
        :return: whether the run is active or not
        """
        return len(self._nodes) != 0

    def get_start(self):
        """
        :return: the start-time of the run
        """
        return self._start

    def get_end(self):
        """
        :return: the end-time of the run
        """
        return self._end

    def get_duration(self):
        """
        :return: the duration of the run
        """
        return self._end - self._start

    def get_total_travel_time(self):
        """
        :return: the travel-duration of the run
        """
        return self._travel_time

    def get_distance(self):
        """
        :return: the duration of the run
        """
        return self._distance

    def get_apparent_gap(self):
        """
        :return: the apparent gap between first and last _runs
        """
        return self.last_request().pick_up_node.earliest_arrival - self.first_request().pick_up_node.earliest_arrival

    def first_request(self):
        """
        :return: the first request in the run
        """
        return self._requests[0]

    def last_request(self):
        """
        :return: the first request in the run
        """
        return self._requests[-1]

    def log_error(self, error, err_type):
        """
        :param error: logs the error
        :param err_type: main type of the error
        """
        if self._log_error:
            if error not in self._error_log:
                self._error_log.append(error)
                self._error_hits[error] = 1
            else:
                self._error_hits[error] += 1
            if err_type in self._error_types_stats.keys():
                self._error_types_stats[err_type] += 1
            else:
                self._error_types_stats[err_type] = 1

    def get_error_log(self):
        """
        :return: error log
        """
        return self._error_log

    def print_error_log(self):
        """
        prints the error log
        """
        for error in self._error_log:
            logger.error(f"{error}, number of hits: {self._error_hits[error]}")

        for error_type in self._error_types_stats.keys():
            logger.error(f"{error_type.value}, number of error counts: {self._error_types_stats[error_type]}")

    def get_time_window_violations(self):
        """
            :return list of time violations
        """
        violations = []
        for key in self._time_window_violations.keys():
            violations.extend(list(set(self._time_window_violations[key])))
        return list(set(violations))

    def get_insertion_violations(self):
        """
            :return list of insertion violations
        """
        violations = []
        for key in self._insertion_violations.keys():
            violations.extend(list(set(self._insertion_violations[key])))
        return list(set(violations))

    def get_capacity_violations(self):
        """
            :return list of capacity violations
        """
        violations = []
        for key in self._capacity_violations.keys():
            violations.extend(list(set(self._capacity_violations[key])))
        return list(set(violations))

    def get_run_violations(self):
        """
            :return list of run-length violations
        """
        violations = []
        for key in self._run_violations.keys():
            violations.append(self._run_violations[key])
        return list(set(violations))

    def get_time_window_violation_for_req(self, request):
        """
            :param request: selected request
            :return time violation for the request
        """
        violations = []
        if request in self._time_window_violations.keys():
            violations.extend(self._time_window_violations[request])
        return list(set(violations))

    def get_insert_violation_for_req(self, request):
        """
            :param request: selected request
            :return time violation for the request
        """
        violations = []
        if request in self._insertion_violations.keys():
            violations.extend(self._insertion_violations[request])
        return list(set(violations))

    def get_capacity_violation_for_req(self, request):
        """
            :param request: selected request
            :return capacity violation for the request
        """
        violations = []
        if request in self._capacity_violations.keys():
            violations.extend(self._capacity_violations[request])
        return list(set(violations))

    def get_run_violation_for_req(self, request):
        """
            :param request: selected request
            :return run-length violation for the request
        """
        violation = None
        if request in self._run_violations.keys():
            violation = self._run_violations[request]
        return violation

    def get_all_violations_for_req(self, request):
        """
        :param request: selected request
        :return: list of violation for the request
        """
        all_violations = []
        if request in self._insertion_violations.keys():
            all_violations.extend(self._insertion_violations[request])
        if request in self._run_violations.keys():
            all_violations.append(self._run_violations[request])
        return all_violations

    def serve_able(self, other_run):
        """
        :return: says whether the baseline run is serve-able
        """
        c1 = self.get_end() + GenDataLoader.instance().agency_config.buffer_time <= other_run.get_start()
        c2 = other_run.get_end() + GenDataLoader.instance().agency_config.buffer_time <= self.get_start()
        is_serve_able = False
        if c1 or c2:
            is_serve_able = True
        return is_serve_able

    @staticmethod
    def get_travel_time(node_x, node_y):
        """
        :param node_x: start node
        :param node_y: end node
        :return: travel time for moving from start node to end node
        """
        _, travel_time = GenDataLoader.instance().get_travel_time(node_x.real_location, node_y.real_location)
        return travel_time

    @staticmethod
    def get_travel_distance(node_x, node_y):
        """
        :param node_x: start node
        :param node_y: end node
        :return: travel distance for moving from start node to end node
        """
        _, distance = GenDataLoader.instance().get_distance(node_x.real_location, node_y.real_location)
        return distance

    def get_prev_follow_reqs(self, prev_node, follow_node):
        """
        :param prev_node: previous node of conflict
        :param follow_node: follow node of conflict
        :return: previous request corresponding with previous node, following request corresponding with
        follow node
        """
        if prev_node != self._cur_request.pick_up_node and prev_node != self._cur_request.drop_off_node:
            prev_req = self._pick_drop_off_map.get_request(prev_node)
        else:
            prev_req = self._cur_request
        if follow_node != self._cur_request.pick_up_node and \
                follow_node != self._cur_request.drop_off_node:
            next_req = self._pick_drop_off_map.get_request(follow_node)
        else:
            next_req = self._cur_request
        return prev_req, next_req

    def adjust_travel_distances(self, nodes):
        """
            :param nodes: nodes in the run
            computing the distance values
        """
        distance_values = [0]
        for i, node in enumerate(nodes):
            if i < len(nodes) - 1:
                # computing travel distance for a node
                travel_distance = self.get_travel_distance(node, nodes[i + 1])
                distance_value = distance_values[i] + travel_distance
                distance_values.append(distance_value)
        return distance_values

    def adjust_values(self, nodes, request, threshold_limit=-1):
        status = True
        new_duration = 0
        total_travel_time = 0
        total_distance = 0
        t_time_src = 0
        t_time_snk = 0
        node_times = {nodes[0].idx: nodes[0].earliest_arrival}
        time_values = [nodes[0].earliest_arrival]
        distance_values = [0]
        dwell_time = GenDataLoader.instance().agency_config.dwell_time
        capacities = [nodes[0].capacity]
        if capacities[0] > GenDataLoader.instance().agency_config.vehicle_capacity:
            if self._log_error:
                cap_v_stat = CapacityViolationStat(
                    self._cur_request, self._cur_request, self._cur_request,
                    nodes[0], nodes[0], self._try_index
                )
                violations = []
                if self._cur_request in self._capacity_violations.keys():
                    violations = self._capacity_violations[self._cur_request]
                violations.append(cap_v_stat)
                self._capacity_violations[self._cur_request] = violations
            status = False
            capacities = []
        else:
            total_travel_time = 0
            for i, node in enumerate(nodes):
                if i < len(nodes) - 1:
                    # first check the TIME-WINDOW constraint

                    # computing travel time for a node
                    travel_time = self.get_travel_time(node, nodes[i + 1])

                    # computing travel distance for a node
                    travel_distance = self.get_travel_distance(node, nodes[i + 1])

                    # total time take to move to the node
                    total_time_taken = travel_time + dwell_time

                    # next time point value
                    time_value = max(time_values[i] + total_time_taken, nodes[i + 1].earliest_arrival)

                    # next distance point value
                    distance_value = distance_values[i] + travel_distance

                    # updating upper limit for drop-off flexibility
                    upper_limit = nodes[i + 1].latest_arrival
                    if nodes[i + 1].node_type == NodeType.DROP_OFF.value:
                        upper_limit += GenDataLoader.instance().agency_config.drop_off_flexibility

                        # checking the maximum detour time
                        pick_up_time = node_times[nodes[i + 1].idx - 1]
                        req = self.get_request(nodes[i + 1])
                        if req is None:
                            req = request
                        if req is not None:
                            detour_time = GenDataLoader.instance().agency_config.max_detour_time
                            if time_value > pick_up_time + dwell_time + req.travel_time + detour_time:
                                status = False
                                time_values = []
                                distance_values = []
                                node_times = {}
                                break

                    if time_value > upper_limit:
                        if self._log_error:
                            actual_gap = upper_limit - time_values[i]
                            prev_req, next_req = self.get_prev_follow_reqs(node, nodes[i + 1])
                            time_v_stat = TimeWindowViolationStat(
                                self._cur_request, travel_time, dwell_time, actual_gap, prev_req, next_req, node,
                                nodes[i + 1], self._try_index
                            )
                            violations = []
                            if self._cur_request in self._time_window_violations.keys():
                                violations = self._time_window_violations[self._cur_request]
                            violations.append(time_v_stat)
                            self._time_window_violations[self._cur_request] = violations
                        status = False
                        time_values = []
                        distance_values = []
                        node_times = {}
                        break
                    elif time_value >= 0:
                        total_travel_time += travel_time
                        time_values.append(time_value)
                        distance_values.append(distance_value)
                        node_times[nodes[i + 1].idx] = time_value
                    else:
                        raise ImplementationError(f"Invalid time value {time_value}")

                    # then check CAPACITY constraint
                    capacity_value = capacities[i] + nodes[i + 1].capacity
                    if capacity_value > GenDataLoader.instance().agency_config.vehicle_capacity:
                        if self._log_error:
                            prev_req, next_req = self.get_prev_follow_reqs(node, nodes[i + 1])
                            cap_v_stat = CapacityViolationStat(
                                self._cur_request, prev_req, next_req, node, nodes[i + 1], self._try_index
                            )
                            violations = []
                            if self._cur_request in self._capacity_violations.keys():
                                violations = self._capacity_violations[self._cur_request]
                            violations.append(cap_v_stat)
                            self._capacity_violations[self._cur_request] = violations
                        status = False
                        capacities = []
                        break
                    elif capacity_value >= 0:
                        capacities.append(capacity_value)
                    else:
                        raise ImplementationError(f"Invalid capacity value {capacity_value}")
        if status:
            if len(time_values) != 2:
                time_values, node_times = self._optimize_times(nodes, time_values, node_times)
            else:
                # for the first requests
                time_values[0] = nodes[0].earliest_arrival
                time_values[1] = nodes[1].earliest_arrival + dwell_time
                node_times[nodes[0].idx] = time_values[0]
                node_times[nodes[1].idx] = time_values[1]

            # finally check RUN-LENGTH constraint
            first = nodes[0]
            last = nodes[-1]
            t_time_src = self.get_travel_time(self._source, first)
            t_time_snk = self.get_travel_time(last, self._sink)
            t_dist_src = self.get_travel_distance(self._source, first)
            t_dist_snk = self.get_travel_distance(last, self._sink)
            dwell_time = GenDataLoader.instance().agency_config.dwell_time

            max_limit = self._max_duration
            if threshold_limit > 0:
                max_limit = threshold_limit
            new_duration = time_values[-1] - time_values[0] + t_time_src + t_time_snk + dwell_time
            total_distance = distance_values[-1] + t_dist_src + t_dist_snk
            if new_duration > max_limit:
                status = False
                time_values = []
                distance_values = []
                node_times = {}
        time_obj = AdjustedTimes(
            status, time_values, node_times, distance_values, t_time_src, t_time_snk,
            new_duration, total_travel_time, total_distance
        )
        capacity_obj = AdjustedCapacities(status, capacities)
        return AdjustedValues(status, time_obj, capacity_obj)

    def _optimize_times(self, nodes, time_values, node_times):
        """
        :param nodes: nodes in the run
        :param time_values: initial time-values by ensuring time-window constraints
        for the ``nodes''
        :return: adjusted travel time values and dictionary of time values for corresponding
        nodes
        """
        # adjusting travel-times to minimize the waiting time
        # in the first step, I start with first node earliest pickup and continue
        # adjusting travel times

        # but that is not actually required, and can introduce wait-times,
        # those wait-times are fixed later once all nodes are assigned

        # this helps to reduce the total run-duration, and improves overall solution
        dwell_time = GenDataLoader.instance().agency_config.dwell_time
        node_rev = nodes[::-1].copy()
        len_v = len(time_values)
        for k, _r_node in enumerate(node_rev):
            k_rev = len_v - k - 1
            if k > 0:
                _prev_travel_time = self.get_travel_time(_r_node, node_rev[k - 1])
                _prev_time = node_times[node_rev[k - 1].idx]
                _cur_time = node_times[_r_node.idx]

                _min_prev_time = _cur_time + _prev_travel_time + dwell_time
                if _min_prev_time < _prev_time:
                    _cur_time = min(_prev_time - _prev_travel_time - dwell_time, _r_node.latest_arrival)
                time_values[k_rev] = _cur_time
                node_times[_r_node.idx] = _cur_time

        for node in nodes:
            flexibility = GenDataLoader.instance().agency_config.drop_off_flexibility
            if not (node.earliest_arrival <= node_times[node.idx] <= node.latest_arrival + flexibility):
                raise ImplementationError("Time-window constraint validation failed")
        return time_values, node_times

    def adjust_capacities(self, nodes):
        """

        This presents to compute the capacities for solutions obtain from other baselines,
        since they only outputs order and time values, this can be also used as an additional verification
        on capacity constraints

        :param nodes: nodes in the run
        :return: adjusted capacities for the run
        """
        status = True
        capacities = [nodes[0].capacity]
        if capacities[0] > GenDataLoader.instance().agency_config.vehicle_capacity:
            if self._log_error:
                cap_v_stat = CapacityViolationStat(
                    self._cur_request, self._cur_request, self._cur_request,
                    nodes[0], nodes[0], self._try_index
                )
                violations = []
                if self._cur_request in self._capacity_violations.keys():
                    violations = self._capacity_violations[self._cur_request]
                violations.append(cap_v_stat)
                self._capacity_violations[self._cur_request] = violations
            status = False
            capacities = []
        else:
            for i, node in enumerate(nodes):
                if i < len(nodes) - 1:
                    capacity_value = capacities[i] + nodes[i + 1].capacity
                    if capacity_value > GenDataLoader.instance().agency_config.vehicle_capacity:
                        if self._log_error:
                            prev_req, next_req = self.get_prev_follow_reqs(node, nodes[i + 1])
                            cap_v_stat = CapacityViolationStat(
                                self._cur_request, prev_req, next_req, node, nodes[i + 1], self._try_index
                            )
                            violations = []
                            if self._cur_request in self._capacity_violations.keys():
                                violations = self._capacity_violations[self._cur_request]
                            violations.append(cap_v_stat)
                            self._capacity_violations[self._cur_request] = violations
                        status = False
                        capacities = []
                        break
                    elif capacity_value >= 0:
                        capacities.append(capacity_value)
                    else:
                        raise ImplementationError(f"Invalid capacity value {capacity_value}")
        return AdjustedCapacities(status, capacities)

    def verify(self):
        response = self.adjust(self._nodes.copy())
        return response.success

    def get_cost(self, assign_ratio=0.0, stat=None):
        """
        :param assign_ratio: previously assigned ratio
        :param stat: computed stats
        :return: cost
        """
        cost = stat.extra_time + (
                self._params["wait_time"] + self._params["assign_fraction_cost"] * assign_ratio
        ) * stat.wait_time
        return cost

    def get_threshold(self, assign_ratio):
        """
        :param assign_ratio: previously assigned ratio
        :return: threshold
        """
        threshold = self._params["score_threshold"] + self._params["assign_fraction_threshold"] * assign_ratio + \
                    self._params["length_of_run"] * self.get_duration()
        return threshold

    def get_placement_indices(self, node, start_index=0):
        """
        this will provide the possible insertion location of the node
        :param node: selected node
        :param start_index: start index
        :return: insertion indices
        """
        return self._get_placement_indices(self._nodes, node, start_index)

    @staticmethod
    def _get_placement_indices(nodes, node, start_index=0):
        """
        this will provide the possible insertion location of the node
        :param node: selected node
        :param start_index: start index
        :return: insertion indices
        """
        n_indices = []
        for i, node_i in enumerate(nodes):
            if i >= start_index:
                # add before all the requests
                if i == 0:
                    if node.reachable(node_i):
                        n_indices.append(0)

                # add at the end of all the requests
                if i == len(nodes) - 1:
                    if node_i.reachable(node):
                        n_indices.append(len(nodes))

                if i < len(nodes) - 1:
                    node_i_next = nodes[i + 1]
                    if node_i.reachable(node) and node.reachable(node_i_next):
                        n_indices.append(i + 1)

        return n_indices

    def adjust(self, temp_nodes, request=None, threshold_limit=-1):
        """
        adjust time and capacity
        :param request: the request that needs to be added
        :param temp_nodes: temporary nodes
        :param threshold_limit: time threshold limit
        :return: return adjust status
        """
        if len(temp_nodes) > 0:
            adjust_value_obj = self.adjust_values(temp_nodes, request, threshold_limit)
        else:
            # this for the case when run becomes empty during merge process
            adjust_value_obj = AdjustedValues(True, None, None)
        return adjust_value_obj

    def adjust_and_compute_stats(self, request, temp_nodes, threshold_limit=-1):
        """
        this will insert both pick-up node and drop-off node
        :param request: the specific request
        :param temp_nodes: temporary nodes
        :param threshold_limit: by enabling this, will allow only duration upto the maximum normal rate
        :return: compute statistics
        """
        response = self.adjust(temp_nodes, request, threshold_limit)
        stat = None
        success = response.success
        if success:
            first = temp_nodes[0]
            last = temp_nodes[-1]
            t_dist_src = self.get_travel_distance(self._source, first)
            t_dist_snk = self.get_travel_distance(last, self._sink)
            extra_time = max(0, response.time_obj.duration - self.get_duration())
            wait_time = 0
            temp_nodes_ml = MappedList(temp_nodes)
            p_idx = temp_nodes_ml.index(request.pick_up_node)
            if len(response.time_obj.times) > 2:
                if p_idx > 0:
                    travel_time = self.get_travel_time(temp_nodes[p_idx - 1], request.pick_up_node)
                    wait_time = request.pick_up_node.earliest_arrival - response.time_obj.times[
                        p_idx - 1] - travel_time - \
                                GenDataLoader.instance().agency_config.dwell_time
                if self._params["consider_neg_wait"] == "False":
                    wait_time = max(0, wait_time)
                    if p_idx != len(temp_nodes) - 3 and p_idx > 0:
                        wait_time = 0
            h = int(request.pick_up_node.earliest_arrival / 3600)
            busyness_pdh = GenDataLoader.instance().demand(pz=request.pick_up_zip, dz=request.drop_off_zip, h=h)
            busyness_ph = GenDataLoader.instance().demand(pz=request.pick_up_zip, h=h)
            busyness_dh = GenDataLoader.instance().demand(dz=request.drop_off_zip, h=h)
            busyness_h = GenDataLoader.instance().demand(h=h)
            tightness = 0
            # computing the tightness of the window
            if 0 < p_idx < len(temp_nodes) - 1:
                sel_n = temp_nodes[p_idx]
                prev_n = temp_nodes[p_idx - 1]
                next_n = temp_nodes[p_idx + 1]
                t_x = max(0, prev_n.latest_arrival - sel_n.earliest_arrival)
                t_y = max(0, sel_n.latest_arrival - next_n.earliest_arrival)
                if t_x + t_y > 0:
                    tightness = abs(t_x - t_y) / (t_x + t_y)
            total_distance = response.time_obj.distances[-1] + t_dist_src + t_dist_snk
            total_travel_time = response.time_obj.travel_time + response.time_obj.src_time + response.time_obj.snk_time
            extra_distance = max(0, total_distance - self.get_distance())
            stat = InsertStat(
                temp_nodes, response.time_obj.times, response.time_obj.node_times, response.capacity_obj.capacities,
                response.time_obj.src_time, response.time_obj.snk_time, total_travel_time, total_distance,
                extra_time, wait_time, busyness_pdh, busyness_ph,
                busyness_dh, busyness_h, tightness, extra_distance
            )
            success = True
        else:
            run_v_stat = RunViolationStat(request, self._try_index)
            if request not in self._run_violations.keys():
                self._run_violations[request] = run_v_stat

        return success, stat

    def adjust_and_compute_stats_force(self, temp_nodes, threshold_limit=-1):
        """
        this will insert both pick-up node and drop-off node
        :param temp_nodes: temporary nodes
        :param threshold_limit: by enabling this, will allow only duration upto the maximum normal rate
        :return: compute statistics, for force insert
        """
        response = self.adjust(temp_nodes, threshold_limit=threshold_limit)
        total_travel_time = response.time_obj.travel_time + response.time_obj.src_time + response.time_obj.snk_time
        success = response.success
        if success:
            stat = InsertStat(
                temp_nodes, response.time_obj.times, response.time_obj.node_times, response.capacity_obj.capacities,
                response.time_obj.src_time, response.time_obj.snk_time, total_travel_time, response.time_obj.distance,
                None, None, None, None, None, None, None, None
            )
        else:
            formatted_time = convert_sec_to_hh_mm_ss(response.time_obj.duration)
            raise ImplementationError(f"Run duration exceeds the limit, current duration {formatted_time}")

        return success, stat

    def adjust_and_compute_stats_force_bw(self, temp_nodes, time_values, total_travel_time, threshold_limit=-1):
        """
        this insert the requests, and maintain the same time
        for baseline wrappers, such as Google OR-tools routing API and VRoom

        :param temp_nodes: temporary nodes
        :param time_values: time values
        :param total_travel_time: total travel time
        :param threshold_limit: by enabling this, will allow only duration upto the maximum normal rate
        :return: compute statistics, for force insert
        """
        capacity_obj = self.adjust_capacities(temp_nodes)
        response_distances = self.adjust_travel_distances(temp_nodes)
        stat = None
        success = capacity_obj.success
        if success:
            node_times_dict = {}
            for i, node in enumerate(temp_nodes):
                flexibility = GenDataLoader.instance().agency_config.drop_off_flexibility
                if not (node.earliest_arrival <= time_values[i] <= node.latest_arrival + flexibility):
                    raise ImplementationError("Time-window constraint validation failed")
                node_times_dict[node.idx] = time_values[i]
            response_times = time_values.copy()
            response_node_times = node_times_dict.copy()
            first = temp_nodes[0]
            last = temp_nodes[-1]
            t_time_src = self.get_travel_time(self._source, first)
            t_time_snk = self.get_travel_time(last, self._sink)
            t_dist_src = self.get_travel_distance(self._source, first)
            t_dist_snk = self.get_travel_distance(last, self._sink)
            dwell_time = GenDataLoader.instance().agency_config.dwell_time
            max_limit = self._max_duration
            if threshold_limit > 0:
                max_limit = threshold_limit
            new_duration = response_times[-1] - response_times[0] + t_time_src + t_time_snk + dwell_time
            total_travel_time = total_travel_time + t_time_src + t_time_snk
            if new_duration <= max_limit:
                total_distance = response_distances[-1] + t_dist_src + t_dist_snk
                stat = InsertStat(
                    temp_nodes, response_times, response_node_times, capacity_obj.capacities,
                    t_time_src, t_time_snk, total_travel_time, total_distance, None, None, None, None,
                    None, None, None, None
                )
                success = True
            else:
                formatted_time = convert_sec_to_hh_mm_ss(new_duration)
                raise ImplementationError(f"Run duration exceeds the limit, current duration {formatted_time}")

        return success, stat

    def insert(self, request, stat):
        """
        this will insert both pick-up node and drop-off node
        :param request: the specific request
        :param stat: this will provide the details nodes after inserting
        """
        self._try_index = 0
        self._travel_time = stat.travel_time
        self._start = stat.times[0] - stat.src_time
        self._end = stat.times[-1] + stat.snk_time
        self._distance = stat.distance
        self._nodes = stat.nodes.copy()
        self._times = stat.times.copy()
        self._node_times = stat.node_times.copy()
        self._capacities = stat.capacities.copy()
        self._pick_drop_off_map.insert_request(request)
        self._requests = []
        if request in self._time_window_violations.keys():
            self._time_window_violations.pop(request)
        if request in self._capacity_violations.keys():
            self._capacity_violations.pop(request)
        if request in self._run_violations.keys():
            self._run_violations.pop(request)
        for node in self._nodes:
            if node.node_type == "pick_up":
                request = self._pick_drop_off_map.get_request(node)
                self._requests.append(request)

    def get(self, node):
        """
        :param node: return the pick-up and drop-off node and associated node,
                     for an example:
                     if the node is a pick-up node then this function provides the corresponding drop-off node as well
                     if the node is a drop-off node then this function provides the corresponding pick-up node as well
        """
        return self._pick_drop_off_map.get(node)

    def get_request(self, node):
        """
        :param node: pick-up or drop-off node
        :return the corresponding request
        """
        return self._pick_drop_off_map.get_request(node)

    def remove(self, node):
        """
        :param node: remove the particular node and associated node,
                     for an example:
                     if the node is a pick-up node then this function remove the corresponding drop-off node
                     if the node is a drop-off node then this function remove the corresponding pick-up node
        :return: whether the particular node is removed or not
        """
        request = self._pick_drop_off_map.get_request(node)
        all_nodes = self._nodes.copy()
        done = 2
        p_idx = -1
        d_idx = -1
        for k, node in enumerate(all_nodes):
            if node.idx == request.pick_up_node.idx:
                p_idx = k
                done -= 1
            elif node.idx == request.drop_off_node.idx:
                d_idx = k
                done -= 1
            if done == 0:
                break
        temp_nodes = all_nodes[:p_idx] + all_nodes[p_idx + 1:d_idx] + all_nodes[d_idx + 1:]
        if len(temp_nodes) > 0:
            response = self.adjust(temp_nodes)
            if not response.success:
                raise ValueError(f"Implementation Logic Error")
            first = temp_nodes[0]
            last = temp_nodes[-1]
            travel_time_src = self.get_travel_time(self._source, first)
            travel_time_snk = self.get_travel_time(last, self._sink)
            self._travel_time = response.time_obj.travel_time + travel_time_src + travel_time_snk
            self._start = response.time_obj.times[0] - travel_time_src
            self._end = response.time_obj.times[-1] + travel_time_snk
            self._nodes = temp_nodes.copy()
            self._times = response.time_obj.times.copy()
            self._node_times = response.time_obj.node_times.copy()
            self._capacities = response.capacity_obj.capacities.copy()
            self._requests.remove(request)
            self._pick_drop_off_map.remove_request(request)
        else:
            self.reset()
        return request

    def remove_request(self, request):
        return self.remove(request.pick_up_node)

    def is_overlap(self, run_x):
        """
        :param run_x: baseline run
        :return: returns whether the run_x, and this run itself overlap in operation
        """
        return not (self._start > run_x.get_end() or self._end < run_x.get_start())

    def reset(self):
        """
            reset the run
        """
        self._start = -1
        self._end = -1
        self._travel_time = 0
        self._nodes = []
        self._requests = []
        self._capacities = []
        self._times = []
        self._node_times = {}
        self._error_log = []
        self._error_hits = {}
        self._error_types_stats = {}
        self._time_window_violations = {}
        self._insertion_violations = {}
        self._capacity_violations = {}
        self._run_violations = {}
        self._cur_request = None
        self._try_index = 0
        self._pick_drop_off_map = RequestDoubleMap()

    def get_max_capacity(self):
        """
        :return: returns the maximum capacity
        """
        return max(self._capacities)

    def get_route(self):
        """
        :return: the route details
        """
        content = f"Run: {self.get_name()} "
        content += f"Start: {convert_sec_to_hh_mm_ss(self._start)} "
        content += f"End: {convert_sec_to_hh_mm_ss(self._end)}\n"
        content += f"Number of request(s): {int(len(self._nodes) / 2)}\n"
        content += f"Entire route detail(s):\n"
        all_nodes = [self._source] + self._nodes + [self._sink]
        for i, node in enumerate(all_nodes[:-1]):
            nxt_nd = all_nodes[i + 1]
            duration = self.get_travel_time(node, nxt_nd)
            content += f"{node.__str__()} -> {nxt_nd.__str__()} {convert_sec_to_hh_mm_ss(duration)}\n"
        content += f"Rider duration(s):\n"
        for k, node in enumerate(self._nodes):
            idx = node.idx
            if idx % 2 == 0 and idx > 0:
                rider_duration = self._node_times[idx] - self._node_times[idx - 1]
                content += f"Request-ID:{idx}, duration: {convert_sec_to_hh_mm_ss(rider_duration)}\n"
        content += f"Total duration: {convert_sec_to_hh_mm_ss(self.get_duration())}\n"
        return content

    @staticmethod
    def get_sub_idx(i):
        """
        :param i: actual index
        :return: substitute index in true locations
        """
        if i + 1 > 2 * GenDataLoader.instance().get_data_size():
            i_sub = 0
        else:
            i_sub = i + 1
        return i_sub

    def get_assign_data(self):
        """
        :return: this will returns the assign data
        """
        all_nodes = [self._source] + self._nodes + [self._sink]
        data = GenDataLoader.instance().data
        assign_data = []
        for _i, _node_i in enumerate(all_nodes[:-1]):
            _j = _i + 1
            _node_j = all_nodes[_j]
            i_sub = self.get_sub_idx(_i)
            j_sub = self.get_sub_idx(_j)
            travel_time = self.get_travel_time(_node_i, _node_j)
            assign_data_entry = GenericAssignData(
                data["true_locations"][i_sub],
                data["time_windows"][_i][0],
                data["time_windows"][_i][1],
                i_sub,
                data["true_locations"][j_sub],
                data["time_windows"][_j][0],
                data["time_windows"][_j][1],
                j_sub,
                travel_time
            )
            assign_data.append(assign_data_entry)
        return assign_data

    def check_feasible(
            self,
            request,
            assign_ratio=0.0,
            threshold_limit=-1,
            is_from_mutation=False
    ):
        raise NotImplementedError

    def check_and_insert(
            self,
            request,
            assign_ratio=0.0,
            threshold_limit=-1,
            is_from_mutation=False
    ):
        """
        this check and insert the request
        :param request: the specific request
        :param assign_ratio: assign ratio
        :param threshold_limit: by enabling this, will allow only duration upto the maximum normal rate
        :param is_from_mutation: this show whether the assignment from mutation
        :return: whether the request is inserted or not
        """
        success, stat = self.check_feasible(
            request=request,
            assign_ratio=assign_ratio,
            threshold_limit=threshold_limit,
            is_from_mutation=is_from_mutation
        )
        if success and stat is not None:
            self.insert(
                request=request,
                stat=stat
            )
        else:
            success = False
        return success

    def insert_force(self, requests, stat):
        """
        this will insert both pick-up node and drop-off node
        :param requests: set of requests
        :param stat: this will provide the details nodes after inserting
        """
        self._try_index = 0
        self._travel_time = stat.travel_time
        self._start = stat.times[0] - stat.src_time
        self._end = stat.times[-1] + stat.snk_time
        self._nodes = stat.nodes.copy()
        self._times = stat.times.copy()
        self._node_times = stat.node_times.copy()
        self._capacities = stat.capacities.copy()
        for request in requests:
            self._pick_drop_off_map.insert_request(request)
            if request in self._time_window_violations.keys():
                self._time_window_violations.pop(request)
            if request in self._capacity_violations.keys():
                self._capacity_violations.pop(request)
            if request in self._run_violations.keys():
                self._run_violations.pop(request)
        self._requests = []
        for node in self._nodes:
            if node.node_type == NodeType.PICK_UP.value:
                request = self._pick_drop_off_map.get_request(node)
                self._requests.append(request)

    def check_and_force_insert(
            self,
            requests,
            temp_nodes,
            threshold_limit=-1,
    ):
        """
        this check and insert the request
        :param requests: set of requests
        :param temp_nodes: insert multiple requests
        :param threshold_limit: by enabling this, will allow only duration upto the maximum normal rate
        :return: whether the request is inserted or not
        """
        success, stat = self.adjust_and_compute_stats_force(temp_nodes, threshold_limit)
        if success and stat is not None:
            self.insert_force(
                requests=requests,
                stat=stat
            )
        else:
            success = False
        return success

    def check_and_force_insert_bw(
            self,
            requests,
            time_values,
            temp_nodes,
            total_travel_time,
            threshold_limit=-1,
    ):
        """
        this check and insert the request for baseline wrappers, such as Google OR-tools routing API and
        VRoom

        :param requests: set of requests
        :param time_values: set of time_values
        :param temp_nodes: insert multiple requests
        :param total_travel_time: total travel time
        :param threshold_limit: by enabling this, will allow only duration upto the maximum normal rate
        :return: whether the request is inserted or not
        """
        success, stat = self.adjust_and_compute_stats_force_bw(
            temp_nodes, time_values, total_travel_time, threshold_limit
        )
        if success and stat is not None:
            self.insert_force(
                requests=requests,
                stat=stat
            )
        else:
            success = False
        return success

    def __str__(self):
        return f"{self.get_name()} {convert_sec_to_hh_mm_ss(self._start), convert_sec_to_hh_mm_ss(self._end)} " \
               f"{convert_sec_to_hh_mm_ss(self.get_duration())}"
