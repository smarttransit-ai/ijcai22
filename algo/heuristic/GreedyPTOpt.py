##################################################################
#                 GREEDY ALGORITHM IMPLEMENTATION                #
##################################################################
#  This is greedy algorithm implementation to solve the custom   #
#  VRP problem with time windows and route-length limitations    #
##################################################################

import math
import sys

from base.model.CSBasePTOpt import CSBasePTOpt
from common.util.common_util import logger, multi_processing_wrapper


class GreedyPTOpt(CSBasePTOpt):
    """
        This class stores greedy solver setup to solve para-transit optimization problem
    """

    def __init__(self, prefix="greedy", run_cls=None, sample_inst_config=None, config=None, custom_args=None):
        super(GreedyPTOpt, self).__init__(prefix, sample_inst_config, config, custom_args)
        self._run_cls = run_cls
        self.__actual_size = len(self.requests)

    def copy(self):
        """
        this function will create the copy of the original instance
        :return: the copy the original object
        """
        greedy_cpy = type(self)(self.prefix, self._run_cls, self.sample_inst_config, self.config, self._args)
        greedy_cpy.source = self.source
        greedy_cpy.sink = self.sink
        greedy_cpy.requests = self.requests.copy()
        greedy_cpy.cost = self.cost
        greedy_cpy.total_travel_time = self.total_travel_time
        greedy_cpy.no_of_requests = self.no_of_requests
        greedy_cpy.no_of_runs = self.no_of_runs
        greedy_cpy.no_of_vehicles = self.no_of_vehicles
        greedy_cpy.run_plans = self.run_plans.copy()
        greedy_cpy.requests = [_req.copy() for _req in self.requests]
        greedy_cpy._runs = [_run.copy() for _run in self._runs]
        greedy_cpy._vehicles = [_veh.copy() for _veh in self._vehicles]
        greedy_cpy._params = self._params.copy()
        greedy_cpy._threshold_limit = self._threshold_limit
        greedy_cpy._assign_runs = self._assign_runs
        greedy_cpy._args = self._args
        greedy_cpy.__actual_size = self.__actual_size
        return greedy_cpy

    def verify(self):
        # before verify make sure the correct filtering applied
        super(GreedyPTOpt, self).verify()

    def solve(self, write_summary=False):
        # before solving make sure the correct filtering applied
        super(GreedyPTOpt, self).solve(write_summary)

    def get_actual_size(self):
        """
            this is the actual size of the problem
        """
        return self.__actual_size

    def assign_requests(self, idx=-1):
        """
        :param idx: index upto which the requests need to be assigned (using serial approach)
        """
        requests = self.get_requests().copy()
        if idx != -1:
            requests = requests[:idx + 1]
        cur_idx = 0
        request_count = len(requests)
        while len(requests) > 0:
            cur_run = self._run_cls(f"run_{cur_idx}", self.source, self.sink, self._params)
            self._runs.append(cur_run)
            requests = self.assign_inner(requests, cur_run)
            cur_idx += 1
            if request_count == len(requests) and request_count > 0:
                logger.error(f"requests can't be assigned further, remaining {request_count}")
                sys.exit(-1)
            else:
                request_count = len(requests)

            if self.sample_inst_config.no_of_runs != -1:
                if cur_idx >= self.sample_inst_config.no_of_runs:
                    logger.error("Not enough runs !!!")
                    sys.exit(-1)

        if len(requests) > 0:
            logger.error(f"Unassigned requests: {len(requests)}")
            sys.exit()
        return len(requests) == 0

    def assign_requests_parallel(self, idx=-1):
        """
        :param idx: index upto which the requests need to be assigned (using parallel approach)
        """
        requests = self.get_requests().copy()
        if idx != -1:
            requests = requests[:idx + 1]
        cur_idx = 0
        request_count = len(requests)
        while len(requests) > 0:
            cur_run = self._run_cls(f"run_{cur_idx}", self.source, self.sink, self._params)
            self._runs.append(cur_run)
            requests = self.assign_inner_parallel(requests, cur_run)
            cur_idx += 1
            if request_count == len(requests) and request_count > 0:
                logger.error(f"requests can't be assigned further, remaining {request_count}")
                sys.exit(-1)
            else:
                request_count = len(requests)

            if self.sample_inst_config.no_of_runs != -1:
                if cur_idx >= self.sample_inst_config.no_of_runs:
                    logger.error("Not enough runs !!!")
                    sys.exit(-1)

        if len(requests) > 0:
            logger.error(f"Unassigned requests: {len(requests)}")
            sys.exit(-1)

        return len(requests) == 0

    def get_by_lowest_weighted_cost_idx(self, requests, selected_run):
        """
        :param requests: set of requests
        :param selected_run: the selected run
        :return: minimum index (using serial computation)
        """
        idx = -1
        min_cost = math.inf
        min_stat = None
        for i, request in enumerate(requests):
            assign_ratio = 1 - len(requests) * 1.0 / len(self.requests)
            success, stat = selected_run.check_feasible(
                request=request, assign_ratio=assign_ratio,
                threshold_limit=self._threshold_limit
            )
            if success:
                cost = selected_run.get_cost(assign_ratio, stat)
                if cost < min_cost:
                    min_cost = cost
                    min_stat = stat
                    idx = i
        return idx, min_stat

    def get_by_lowest_weighted_cost_idx_parallel(self, requests, selected_run):
        """
        :param requests: set of requests
        :param selected_run: the selected run
        :return: minimum index (using parallel computation)
        """
        idx = -1
        min_cost = math.inf
        min_stat = None
        arguments = []
        for i, request in enumerate(requests):
            indices = selected_run.check_feasible_parallel(
                request=request, threshold_limit=self._threshold_limit
            )
            arguments.extend([[request, p, d, i, self._threshold_limit] for p, d in indices])

        if len(arguments) > 0:

            results = multi_processing_wrapper(
                selected_run.compute_stat_parallel, arguments, int(self._args.no_of_workers),
                wait_for_response=True, skip_logging=True
            )

            assign_ratio = 1 - len(requests) * 1.0 / len(self.requests)
            for (success, stat, k) in results:
                if success:
                    cost = selected_run.get_cost(assign_ratio, stat)
                    threshold = selected_run.get_threshold(assign_ratio)
                    if (cost < min_cost and cost < threshold) or \
                            (cost < min_cost and len(selected_run.get_nodes()) == 0):
                        min_cost = cost
                        min_stat = stat
                        idx = k
        return idx, min_stat

    def assign_inner(self, requests, cur_run):
        """
            finding the request with minimum cost using serial computations
        """
        while len(requests) > 0:
            s_idx, s_stat = self.get_by_lowest_weighted_cost_idx(requests, cur_run)
            if s_idx != -1:
                request = requests[s_idx]
                cur_run.insert(request=request, stat=s_stat)
                requests.remove(request)
            else:
                break
        return requests

    def assign_inner_parallel(self, requests, cur_run):
        """
            finding the request with minimum cost using parallel computations
        """
        while len(requests) > 0:
            s_idx, s_stat = self.get_by_lowest_weighted_cost_idx_parallel(requests, cur_run)
            if s_idx != -1:
                request = requests[s_idx]
                cur_run.insert(request=request, stat=s_stat)
                requests.remove(request)
            else:
                break
        return requests
