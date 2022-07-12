##########################################################
#             RUN-IMPLEMENTATION FOR GREEDY              #
##########################################################

import math
import random

from base.data.GenDataLoader import GenDataLoader
from base.entity.AbstractRun import AbstractRun
from base.errors import ImplementationError


class Run(AbstractRun):
    def check_feasible(
            self,
            request,
            assign_ratio=0.0,
            threshold_limit=-1,
            is_from_mutation=False
    ):
        """
        this will insert both pick-up node and drop-off node
        this follow DETERMINISTIC GREEDY
        :param request: the specific request
        :param assign_ratio: assign ratio
        :param threshold_limit: by enabling this, will allow only duration upto the maximum normal rate
        :param is_from_mutation: this show whether the assignment from mutation
        :return: whether the particular pick-up node and drop-off node can be inserted or not, and status of
        insertion of request if the request can be inserted
        """
        min_stat = None
        self._cur_request = request
        max_duration = self._max_duration
        pick_up_node = request.pick_up_node
        drop_off_node = request.drop_off_node
        if threshold_limit > 0:
            max_duration = threshold_limit
        if len(self._nodes) == 0:
            temp_nodes = [pick_up_node, drop_off_node]
            success, min_stat = self.adjust_and_compute_stats(request, temp_nodes, threshold_limit)
        elif drop_off_node.earliest_arrival < self._start + max_duration:
            p_indices = self.get_placement_indices(pick_up_node)
            d_indices = []
            if len(p_indices) > 0:
                d_indices = self.get_placement_indices(drop_off_node, start_index=min(p_indices) - 1)
            if len(p_indices) > 0 and len(d_indices) > 0:
                min_cost = math.inf
                min_stat = None
                for p_idx in p_indices:
                    for d_idx in d_indices:
                        if p_idx <= d_idx:
                            temp_nodes = self._nodes[:p_idx] + [pick_up_node] + self._nodes[p_idx:d_idx] + \
                                         [drop_off_node] + self._nodes[d_idx:]
                            success, stat = self.adjust_and_compute_stats(request, temp_nodes, threshold_limit)
                            if success:
                                cost = self.get_cost(assign_ratio, stat)
                                threshold = self.get_threshold(assign_ratio)
                                if is_from_mutation:
                                    if cost < min_cost:
                                        min_cost = cost
                                        min_stat = stat
                                else:
                                    if cost < min_cost and cost < threshold:
                                        min_cost = cost
                                        min_stat = stat

        if min_stat is None:
            success = False
        else:
            success = True
        return success, min_stat

    def check_feasible_parallel(
            self,
            request,
            threshold_limit=-1,
    ):
        """
        this will insert both pick-up node and drop-off node
        this follow DETERMINISTIC GREEDY
        :param request: the specific request
        :param threshold_limit: by enabling this, will allow only duration upto the maximum normal rate
        :return: whether the particular pick-up node and drop-off node can be inserted or not, and status of
        insertion of request if the request can be inserted
        """
        self._cur_request = request
        max_duration = self._max_duration
        pick_up_node = request.pick_up_node
        drop_off_node = request.drop_off_node
        indices = []
        if threshold_limit > 0:
            max_duration = threshold_limit
        if len(self._nodes) == 0:
            indices.append([0, 0])
        elif drop_off_node.earliest_arrival < self._start + max_duration:
            p_indices = self.get_placement_indices(pick_up_node)
            d_indices = []
            if len(p_indices) > 0:
                d_indices = self.get_placement_indices(drop_off_node, start_index=min(p_indices) - 1)
            if len(p_indices) > 0 and len(d_indices) > 0:
                for p_idx in p_indices:
                    for d_idx in d_indices:
                        if p_idx <= d_idx:
                            indices.append([p_idx, d_idx])
        return indices

    def get_feasible_indices(
            self,
            request,
            threshold_limit=-1,
    ):
        """
        :param request: the specific request
        :param threshold_limit: by enabling this, will allow only duration upto the maximum normal rate
        :return: whether the particular pick-up node and drop-off node can be inserted or not, and status of
        insertion of request if the request can be inserted
        """
        feasible_indices = []
        min_stats = []
        self._cur_request = request
        max_duration = self._max_duration
        pick_up_node = request.pick_up_node
        drop_off_node = request.drop_off_node
        if threshold_limit > 0:
            max_duration = threshold_limit
        if len(self._nodes) == 0:
            temp_nodes = [pick_up_node, drop_off_node]
            success, min_stat = self.adjust_and_compute_stats(request, temp_nodes, threshold_limit)
            feasible_indices.append([0, 0])
            min_stats.append(min_stat)
        elif drop_off_node.earliest_arrival < self._start + max_duration:
            p_indices = self.get_placement_indices(pick_up_node)
            d_indices = []
            if len(p_indices) > 0:
                d_indices = self.get_placement_indices(drop_off_node, start_index=min(p_indices) - 1)
            if len(p_indices) > 0 and len(d_indices) > 0:
                for p_idx in p_indices:
                    for d_idx in d_indices:
                        if p_idx <= d_idx:
                            temp_nodes = self._nodes[:p_idx] + [pick_up_node] + self._nodes[p_idx:d_idx] + \
                                         [drop_off_node] + self._nodes[d_idx:]
                            success, stat = self.adjust_and_compute_stats(request, temp_nodes, threshold_limit)
                            if success:
                                feasible_indices.append([p_idx, d_idx])
                                min_stats.append(stat)

        return feasible_indices, min_stats

    def compute_stat_parallel(self, arguments):
        request, p_idx, d_idx, k, threshold_limit = arguments
        pick_up_node = request.pick_up_node
        drop_off_node = request.drop_off_node
        temp_nodes = self._nodes[:p_idx] + [pick_up_node] + self._nodes[p_idx:d_idx] + \
                     [drop_off_node] + self._nodes[d_idx:]
        success, stat = self.adjust_and_compute_stats(request, temp_nodes, threshold_limit)
        return success, stat, k

    def merge(self, other_runs, threshold_limit=-1):
        """
        This function will try to merge one run with another run
        without violating the operational constraints
        :param other_runs: baseline runs, it can be either single run, or list of runs
        :param threshold_limit: threshold limit
        :return: says whether a run is merged or not, and merged run
        """
        self_copy = self.copy()
        if isinstance(other_runs, Run):
            other_runs = [other_runs]
        other_runs_copy = other_runs.copy()
        other_start_times = [other.get_start() for other in other_runs_copy]
        other_end_times = [other.get_end() for other in other_runs_copy]
        is_merged = False
        min_start = min(self_copy.get_start(), min(other_start_times))
        max_end = max(self_copy.get_end(), max(other_end_times))
        count = 0
        for run in other_runs_copy:
            count += run.get_size()
        if max_end - min_start <= self._max_duration:
            success = False
            for other_run in other_runs_copy:
                other_run_requests = other_run.get_requests()
                for request in other_run_requests:
                    success = self_copy.check_and_insert(request=request, is_from_mutation=True)
                    if success:
                        count -= 1
                    else:
                        break
                if not success:
                    break
            max_duration = GenDataLoader.instance().agency_config.run_max_duration
            if threshold_limit > 0:
                max_duration = threshold_limit
            if success and count == 0 and self_copy.get_duration() <= max_duration:
                is_merged = True
            elif success and count != 0:
                raise ImplementationError("Error in implementation of Run.merge()")
            else:
                self_copy = self.copy()
        return self_copy, is_merged

    def split_and_merge(self, other_runs, fixed_pattern=True, threshold_limit=-1):
        """
        This function will tries to split the run and merge one run with another run
        without violating the operational constraints
        :param other_runs: second run, or baseline _runs
        :param fixed_pattern: this will make a fix pattern
        :param threshold_limit: threshold limit
        :return: split and merge the runs with status
        """
        success = False
        if isinstance(other_runs, Run):
            other_runs = [other_runs]
        all_runs = [self] + other_runs
        total_runs = len(all_runs)
        x_splits = [[] for _ in range(total_runs)]
        count = 0
        for i, sel_run in enumerate(all_runs):
            x_split_values, success = sel_run.split_n(n=total_runs)
            if not success:
                break
            x_splits[i] = x_split_values
            count += sum([x_split.get_size() for x_split in x_splits[i]])

        if success:
            shuffle_values = []
            remaining_lists = [[k for k in range(total_runs)] for _ in range(total_runs - 1)]
            for i in range(total_runs):
                if not fixed_pattern:
                    shuffle_array = [random.choice(remaining_list) for remaining_list in remaining_lists]
                    for r, remaining_list in enumerate(remaining_lists):
                        remaining_list.remove(shuffle_array[r])
                else:
                    shuffle_array = [((i + k + 1) % total_runs) for k in range(total_runs - 1)]
                shuffle_values.append(shuffle_array)

            merged_runs = []
            merged_status = []
            for k in range(total_runs):
                m_run, m_status = x_splits[0][k].merge([x_splits[i][shuffle_values[k][i - 1]]
                                                        for i in range(1, total_runs)],
                                                       threshold_limit=threshold_limit)
                if not m_status:
                    merged_runs = []
                    merged_status = [False]
                    break
                else:
                    count -= m_run.get_size()
                    merged_runs.append(m_run)
                    merged_status.append(m_status)
            if count != 0 and all(merged_status):
                raise ImplementationError("Implementation error in Run.split_and_merge()")
        else:
            merged_runs = []
            merged_status = [False]
        return merged_runs, all(merged_status)

    def split_n(self, n=2):
        """
        This function will tries to split one run into n _runs
        without violating the operational constraints
        :param n: the number of splits that has to be made to the run
        :return: the run as list of splits
        """
        success = False
        if n >= 2:
            part_runs = []
            self_copy = self.copy()
            for i in range(n):
                run = Run(f"{self._name}s{i}", self._source, self._sink, self._params)
                part_runs.append(run)

            duration = int(self.get_duration() / n) + 1
            run_x_nodes = self_copy.get_nodes()
            split_nodes = {}
            split_requests = {}

            for run_x_node in run_x_nodes:
                request = self_copy.get_request(run_x_node)
                p_node = request.pick_up_node
                for k in range(n):
                    add_in_k = False
                    if k == 0 and p_node.earliest_arrival < self._start + duration:
                        add_in_k = True
                    elif k == n - 1 and self._start + duration * (n - 1) <= p_node.earliest_arrival:
                        add_in_k = True
                    elif self._start + duration * k <= p_node.earliest_arrival < self._start + duration * (k + 1):
                        add_in_k = True
                    if add_in_k:
                        if k not in split_nodes.keys():
                            split_nodes[k] = []
                        split_nodes[k].append(run_x_node)
                        if k not in split_requests.keys():
                            split_requests[k] = []
                        if p_node == run_x_node:
                            split_requests[k].append(request)

            for _k in split_nodes.keys():
                requests = split_requests[_k]
                temp_nodes = split_nodes[_k]
                success = part_runs[_k].check_and_force_insert(requests=requests, temp_nodes=temp_nodes)
                if len(part_runs[_k].get_requests()) != len(requests):
                    part_runs = []
                    success = False
                    break
                else:
                    part_runs[_k].verify()
        else:
            raise ImplementationError("Split should be greater than or equal to 2")
        return part_runs, success
