##################################################################
#                  ONLINE BOOKING SIMULATOR                      #
##################################################################
#  In this mixed optimizer implementation, first the algorithm   #
#  obtain the optimal tight pickup windows from broader pickup   #
#  windows, then search for optimal solution using greedy and    #
#  simulated annealing algorithm                                 #
##################################################################

from ast import literal_eval
from collections import namedtuple
from enum import Enum

from base.common.util import sim_anneal_search
from base.data.GenDataLoader import GenDataLoader
from common.constant.constants import AGENCY_ITER_DIR, RESULT_DIR
from common.util.common_util import logger, multi_processing_wrapper, create_dir, file_exists

SolutionStat = namedtuple('SolutionStat', ['solution', 'alpha'])


class MSModes(Enum):
    SIMPLE_RL_AGENT = 0


def convert_stat_to_feature_vec(stat, exp_req, accept_positions=None, zero_positions=None):
    temp = [
        stat.busyness_pdh, stat.busyness_ph, stat.busyness_dh, stat.busyness_h,
        stat.extra_time, stat.extra_distance, stat.tightness, exp_req
    ]
    updated_vec = []

    if isinstance(accept_positions, str):
        accept_positions = literal_eval(accept_positions)
    if accept_positions is not None and len(accept_positions) > 0:
        for pos in accept_positions:
            if pos < len(temp):
                updated_vec.append(temp[pos])
    else:
        updated_vec = temp.copy()

    if isinstance(zero_positions, str):
        zero_positions = literal_eval(zero_positions)
    if zero_positions is not None and len(zero_positions) > 0:
        for pos in zero_positions:
            if pos < len(updated_vec):
                updated_vec[pos] = 0
    return updated_vec


class OnlineBookingSimulator(object):
    """
        This class provides a wrapper implementation to the online booking simulation, which incorporates following
        1. Determining tight-pickup windows in real-time during booking
        2. Running anytime algorithm in between any two such booking
    """

    def __init__(
            self,
            run_cls,
            rt_solver_cls,
            sample_inst_config=None,
            config=None,
            custom_args=None,
            custom_agency_config=None
    ):
        # agent_with_anytime solver always works in real-time mode
        custom_args.real_time_mode = True
        self._run_cls = run_cls
        self._sample_inst_config = sample_inst_config
        self._config = config
        self._args = custom_args
        self._agency_config = custom_agency_config
        self._custom_anytime_solver_cls = rt_solver_cls
        summary_file_common = f"sa_t{self._sample_inst_config.no_of_trips}_" + \
                              f"gv{self._sample_inst_config.no_of_runs}_" + \
                              f"ev{self._sample_inst_config.no_of_vehicles}_" + \
                              f"{self._sample_inst_config.random_seed}"
        self.summary_iter_file_name = f"{AGENCY_ITER_DIR}/{summary_file_common}"
        self.summary_iter_file_name = self.summary_iter_file_name.format(
            self._sample_inst_config.selected_agency, self._sample_inst_config.selected_date
        )

        # set the total number of seconds the solver will run entire simulation
        if self._args is not None:
            self.total_plan_duration = int(self._args.plan_duration)
            self.anytime_plan_duration = int(self._args.anytime_plan_duration)
            self.prefix = self._args.algo
        else:
            self.total_plan_duration = 300
            self.anytime_plan_duration = 300
            self.prefix = "agent_with_anytime"

        # create the initial solution, (this is just sample instances initialization),
        # though, by default it will take the implementation of GreedyPTOpt (algo/heuristic/GreedyPTOpt)
        # it can be used to baseline additional functionalities

        # in-case if you want to modify anything, please feel free to modify GreedyPTOpt (algo/heuristic/GreedyPTOpt)
        self._cur_sol = self._create_custom_anytime_solver()
        self._cur_sol.set_bw_at_init()

        # at the start, we don't solve the problem so the cost is None
        self._cur_cost = None

        # at the start, we get the copy the all requests,
        # node this is just to keep as attribute only, when solving we will consider request
        # one by one based on the booking id of the requests
        self.requests = self._cur_sol.get_requests().copy()

        # parameters for that can determine the costs for the deterministic greedy and real-time greedy
        self._params = {}
        self.set_params()

        # this will indicates the index of next request need to be assigned, this value get incremented
        # when ever the new requests get assigned to a run the simulation, and allow next request to loaded
        # in to the system.
        self._cur_req_idx = 0

        self._cur_stats = None

        # at the start of the day, the planning hour, is the time at which the agency starts
        # planning for the day
        self._cur_planning_hour = GenDataLoader.instance().agency_config.planning_start_hour

        # this indicates the full state
        self._cur_full_state = self.get_state()

        # these two are for speeding up the training process
        # cost before selecting the time-window
        self._cost_before = -1

        # cost after selecting the time-window
        self._cost_after = -1

        # this used to keep track of the optimal solution of the system
        self._min_sol = self._cur_sol

        # this used to keep track of the optimal cost of the system
        self._min_cost = None

        start = GenDataLoader.instance().agency_config.planning_start_hour

        # this states the expected request for the day
        self._exp_req = GenDataLoader.instance().exp_req(start)

        # this additional configuration, which only solves the real-time problem, without simulated annealing search
        self.only_real_time = False

        # this choose the mode of operation
        self.mixed_mode = MSModes.SIMPLE_RL_AGENT

    def _create_custom_anytime_solver(self):
        custom_anytime_solver = self._custom_anytime_solver_cls(
            prefix=self.prefix,
            run_cls=self._run_cls,
            sample_inst_config=self._sample_inst_config,
            config=self._config,
            custom_args=self._args
        )
        return custom_anytime_solver

    def set_params(self):
        if self._args is not None:
            str_params = ["consider_neg_wait"]
            float_params = [
                # for deterministic greedy
                "wait_time", "assign_fraction_cost", "score_threshold",
                "assign_fraction_threshold", "length_of_run"

                # for non-deterministic greedy
                                             "soft_max_param"
            ]
            for str_param in str_params:
                if str_param in self._args.__dict__.keys():
                    self._params[str_param] = str(self._args.__dict__[str_param])

            for float_param in float_params:
                if float_param in self._args.__dict__.keys():
                    self._params[float_param] = float(self._args.__dict__[float_param])

    def reset(self):
        """
            This function will reset the entire simulation framework,
            this will start the solver from the day of planning.
        """
        self._cur_sol = self._create_custom_anytime_solver()
        self._cur_sol.set_bw_at_init()
        self.requests = self._cur_sol.get_requests().copy()
        self._cur_planning_hour = GenDataLoader.instance().agency_config.planning_start_hour
        self._cur_req_idx = 0
        self._cost_before = -1
        self._cost_after = -1
        self._cur_stats = None
        self._cur_cost = None
        self._min_sol = self._cur_sol
        self._min_cost = None
        self.set_params()
        self._cur_full_state = self.get_state()

    def get_runs(self):
        """
            This function will provide the current runs,
            it can is a list of runs of Run object (base/entity/Run)
        """
        return self._cur_sol.get_runs()

    def current_req(self):
        """
            This function will provide the current request
        """
        return self.requests[self._cur_req_idx]

    def get_planning_hour(self):
        """
            This function will return the current planning hour (it can be in float)
        """
        self.update_planning_hour()
        return self._cur_planning_hour

    def get_all_requests(self):
        """
            This function returns the all the requests in the system
        """
        return self.requests

    def modify_params(self, params):
        """
            set the parameters defined in :param params
        """
        self._params.update(params.copy())

    def update_planning_hour(self):
        """
            update the planning hour
        """
        start = GenDataLoader.instance().agency_config.planning_start_hour
        end = GenDataLoader.instance().agency_config.planning_end_hour
        diff = end - start
        current_plan_ratio = (self._exp_req - self._cur_req_idx) / self._exp_req
        self._cur_planning_hour = start + current_plan_ratio * diff

    def get_state(self):
        """
            This is a dummy implementation of the state
        """
        if self._cur_req_idx < len(self.requests):
            try:
                features, targets = self.get_feasible_combinations_parallel()
            except AssertionError:
                # when the main process is parallel make the sub process as sequential
                # though this is a bad fix, but save lots of time !!!
                features, targets = self.get_feasible_combinations()
            self._cur_full_state = features, targets
        return self._cur_full_state

    @staticmethod
    def compute_cost(solution):
        """
        :param solution: sample solution
        compute the cost and return the cost
        """
        solution_cpy = solution.copy()
        try:
            solution_cpy.solve_parallel()
        except AssertionError:
            # when the main process is parallel make the sub process as sequential
            # though this is a bad fix, but save lots of time !!!
            solution_cpy.solve()
        return solution_cpy.compute_cost()

    @staticmethod
    def specify_req(request, alpha):
        """
        :param request: incoming request
        :param alpha: the earliest pickup time
        :return: request with pick-up and drop-off window adjusted
        """
        step_size = GenDataLoader.instance().agency_config.time_window_gap
        travel_time = request.travel_time
        request.pick_up_node.earliest_arrival = alpha
        request.pick_up_node.latest_arrival = alpha + step_size
        request.drop_off_node.earliest_arrival = alpha + travel_time
        request.drop_off_node.latest_arrival = alpha + step_size + travel_time
        return request

    def get_feasible_combinations(self):
        """
            Get the feasible combinations
        """
        cur_sol_cpy = self._cur_sol.copy()
        cur_sol_runs = cur_sol_cpy.get_runs()
        request = self.current_req()
        alpha, beta = request.broad_time_windows
        step_size = GenDataLoader.instance().agency_config.time_window_gap
        full_indices = []
        full_stats = {}
        for w_i in range(int((beta - alpha) / step_size)):
            w = alpha + w_i * step_size
            request = self.specify_req(request, w)
            for i, sel_run in enumerate(cur_sol_runs):
                indices, stats = sel_run.get_feasible_indices(
                    request=request
                )
                full_indices.extend([[w_i, i, p, d] for (p, d) in indices])
                for k in range(len(stats)):
                    full_stats[w_i, i, indices[k][0], indices[k][1]] = stats[k]

        if len(full_indices) == 0:
            for w_i in range(int((beta - alpha) / step_size)):
                w = alpha + w_i * step_size
                request = self.specify_req(request, w)
                cur_run = self._run_cls(
                    f"run_{len(cur_sol_runs)}", cur_sol_cpy.source, cur_sol_cpy.sink, self._params
                )
                n_indices, n_stats = cur_run.get_feasible_indices(
                    request=request
                )
                full_indices.append([w_i, len(cur_sol_runs), 0, 0])
                full_stats[w_i, len(cur_sol_runs), n_indices[0][0], n_indices[0][1]] = n_stats[0]
        features = []
        targets = []
        exp_req = GenDataLoader.instance().exp_req(h=self._cur_planning_hour)
        for (w, r, p, d) in full_stats.keys():
            features.append(
                convert_stat_to_feature_vec(
                    full_stats[(w, r, p, d)], exp_req, self._args.features, self._args.zero_positions
                )
            )
            targets.append([w, r, p, d])
        self._cur_stats = full_stats
        return features, targets

    def get_feasible_combinations_parallel(self):
        """
            Get the feasible combinations
        """
        cur_sol_cpy = self._cur_sol.copy()
        cur_sol_runs = cur_sol_cpy.get_runs()
        request = self.current_req()
        full_indices = []
        full_stats = {}
        alpha, beta = request.broad_time_windows
        step_size = GenDataLoader.instance().agency_config.time_window_gap
        arguments = [w_i for w_i in range(int((beta - alpha) / step_size))]

        responses = multi_processing_wrapper(
            self.get_feasible_parallel_support, arguments, int(self._args.no_of_workers),
            wait_for_response=False, skip_logging=True
        )

        for response in responses:
            run_indices, run_stats = response
            full_indices += run_indices
            full_stats.update(run_stats.copy())

        if len(full_indices) == 0:
            for w_i in range(int((beta - alpha) / step_size)):
                w = alpha + w_i * step_size
                request = self.specify_req(request, w)
                cur_run = self._run_cls(
                    f"run_{len(cur_sol_runs)}", cur_sol_cpy.source, cur_sol_cpy.sink, self._params
                )
                n_indices, n_stats = cur_run.get_feasible_indices(
                    request=request
                )
                full_indices.append([w_i, len(cur_sol_runs), 0, 0])
                full_stats[w_i, len(cur_sol_runs), n_indices[0][0], n_indices[0][1]] = n_stats[0]
        features = []
        targets = []
        exp_req = GenDataLoader.instance().exp_req(h=self._cur_planning_hour)
        for (w, r, p, d) in full_stats.keys():
            features.append(
                convert_stat_to_feature_vec(
                    full_stats[(w, r, p, d)], exp_req, self._args.features, self._args.zero_positions
                )
            )
            targets.append([w, r, p, d])
        self._cur_stats = full_stats
        return features, targets

    def get_feasible_parallel_support(self, w_i):
        """
        THIS USES A PARALLEL MODE,
        WHICH CAN PROVIDE SOLUTION SO QUICKLY THAN NORMAL MODE
        :param w_i: window_index
        :return: provide feasibility specific time window
        """
        cur_sol_cpy = self._cur_sol.copy()
        cur_sol_runs = cur_sol_cpy.get_runs()
        request = self.current_req()
        alpha, beta = request.broad_time_windows
        step_size = GenDataLoader.instance().agency_config.time_window_gap
        w = alpha + w_i * step_size
        request = self.specify_req(request, w)
        run_indices = []
        run_stats = {}
        for i, sel_run in enumerate(cur_sol_runs):
            indices, stats = sel_run.get_feasible_indices(
                request=request
            )
            run_indices.extend([[w_i, i, p, d] for (p, d) in indices])
            for k in range(len(stats)):
                run_stats[w_i, i, indices[k][0], indices[k][1]] = stats[k]

        return run_indices, run_stats

    def optimal_time_window_rl(self, action):
        """
        This function can be used for reinforcement learning
        :param action: action
        :return: optimal solution and optimal earliest pick-up time
        """
        done = False
        request = self.requests[self._cur_req_idx]
        alpha, beta = request.broad_time_windows
        step_size = GenDataLoader.instance().agency_config.time_window_gap
        w_idx, run_idx, p_idx, d_idx = action
        i = alpha + w_idx * step_size
        if self._cost_before == -1:
            self._cost_before = self.compute_cost(self._cur_sol)
        request = self.specify_req(request, i)
        self._cur_sol.update_request(request, self._cur_req_idx)
        solution = self.insert_request(request, w_idx, run_idx, p_idx, d_idx)
        solution.compute_cost()
        self._cost_after = self.compute_cost(solution)
        cost_diff = self._cost_before - self._cost_after
        self._cost_before = self._cost_after
        min_sol = SolutionStat(solution, i)
        self._cur_sol = solution.copy()
        self._min_sol = solution.copy()
        self._cur_req_idx += 1
        self.update_planning_hour()
        if self._cur_req_idx == len(self.requests):
            self._cur_sol.assign_runs()
            self._cur_sol.write_solution()
            done = True
        return done, min_sol, cost_diff

    def optimal_time_window_rl_with_sa(self, action):
        """
        This function can be used for reinforcement learning,
        with simulated annealing

        :param action: action
        :return: optimal solution and optimal earliest pick-up time
        """
        done = False
        request = self.requests[self._cur_req_idx]
        alpha, beta = request.broad_time_windows
        step_size = GenDataLoader.instance().agency_config.time_window_gap
        w_idx, run_idx, p_idx, d_idx = action
        i = alpha + w_idx * step_size
        if self._cost_before == -1:
            self._cost_before = self.compute_cost(self._cur_sol)
        request = self.specify_req(request, i)
        self._cur_sol.update_request(request, self._cur_req_idx)
        solution = self.insert_request(request, w_idx, run_idx, p_idx, d_idx)
        cur_sol = solution.copy()
        _, _, cur_sol, solution_cost, _ = self._run_with_anytime(cur_sol, i)
        self._cost_after = self.compute_cost(cur_sol)
        cost_diff = self._cost_before - self._cost_after
        self._cost_before = self._cost_after
        min_sol = SolutionStat(solution, i)
        self._cur_sol = cur_sol
        self._min_sol = cur_sol
        self._cur_req_idx += 1
        self.update_planning_hour()
        if self._cur_req_idx == len(self.requests):
            self._cur_sol.assign_runs()
            self._cur_sol.write_solution()
            done = True
        return done, min_sol, cost_diff

    def insert_request(self, request, w_idx, run_idx, p_idx, d_idx):
        stat = self._cur_stats[w_idx, run_idx, p_idx, d_idx]
        cur_sol_cpy = self._cur_sol.copy()
        if run_idx >= len(cur_sol_cpy.get_runs()):
            cur_run = self._run_cls(
                f"run_{run_idx}", cur_sol_cpy.source, cur_sol_cpy.sink, self._params
            )
            cur_sol_cpy.add_run(cur_run)
        else:
            cur_run = cur_sol_cpy.get_runs()[run_idx]

        cur_run.insert(
            request=request,
            stat=stat
        )
        return cur_sol_cpy

    def optimal_time_window_trained_rl(self, idx=-1):
        """
        :return: optimal solution and optimal earliest pick-up time, run idx, pick-up idx, drop-off idx
        """
        from base.learn.RealTimeRLAgent import RealTimeRLAgentNN
        if not isinstance(self._args.features, list):
            lst = literal_eval(self._args.features)
        else:
            lst = self._args.features
        size = len(lst)
        if size == 0:
            size = 8
        agent_model = str(self._args.train_agent_model)
        ds = "neural_network"
        agent_class = RealTimeRLAgentNN
        agent = agent_class(state_space=size)
        agency = self._sample_inst_config.selected_agency
        if idx <= 0:
            if isinstance(agent, RealTimeRLAgentNN):
                model_file_name = f"data/{agency}/models/{self.prefix}_model.h5"
                weight_file_name = f"data/{agency}/models/{self.prefix}_weights.h5"
                agent.load_model(model_file_name, weight_file_name)
        else:
            rs = int(self._args.random_seed)
            fe_status = False
            while not fe_status and idx >= 0:
                if agent_model == "nn":
                    fe_status = file_exists(
                        f"data/{agency}/trainings/{ds}/result_{rs}/{self.prefix}_model_{idx}", ".h5"
                    )
                idx = idx - 1

            if fe_status:
                model_file_name = f"data/{agency}/trainings/{ds}/result_{rs}/{self.prefix}_model_{idx}.h5"
                weight_file_name = f"data/{agency}/trainings/{ds}/result_{rs}/{self.prefix}_weights_{idx}.h5"
                agent.load_model(model_file_name, weight_file_name)
            else:
                raise FileNotFoundError(f"No model exists for model instance {rs}")
        request = self.requests[self._cur_req_idx]
        alpha, beta = request.broad_time_windows
        step_size = GenDataLoader.instance().agency_config.time_window_gap
        _, action = agent.act_eval(self.get_state())
        w_idx, run_idx, p_idx, d_idx = action
        i = alpha + w_idx * step_size
        request = self.specify_req(request, i)
        self._cur_sol.update_request(request, self._cur_req_idx)
        solution = self.insert_request(request, w_idx, run_idx, p_idx, d_idx)
        return SolutionStat(solution, i)

    def solve(self, write_summary=False):
        """
            SOLVE THE ENTIRE PROBLEM
        """
        if self.only_real_time:
            # this is running real-time problem only
            for i, request in enumerate(self.requests):
                # obtain the optimal time-window
                optimal_sol = self.optimal_time_window_trained_rl(idx=int(self._args.model_index))

                # update the request with optimal time-window
                self.specify_req(request, optimal_sol.alpha)
                self._cur_sol.update_request(request, i)
                self._cur_sol = optimal_sol.solution.copy()
                self._cur_cost = self._cur_sol.compute_cost()

                self._cur_req_idx += 1
                self.update_planning_hour()

            self._cur_sol.assign_runs()
            self._cur_cost = self._cur_sol.compute_cost()
            self._min_sol = self._cur_sol
            self._min_cost = self._cur_cost
        else:
            # this is running real-time problem, greedy and simulated annealing search
            for i, request in enumerate(self.requests):
                # obtain the optimal time-window
                optimal_sol = self.optimal_time_window_trained_rl(idx=int(self._args.model_index))

                # update the request with optimal time-window
                self.specify_req(request, optimal_sol.alpha)
                self._cur_sol.update_request(request, i)
                self._cur_sol = optimal_sol.solution.copy()

                _, _, min_sol, min_cost, _ = self._run_with_anytime(self._cur_sol, i)

                self._min_sol = min_sol
                self._min_cost = min_cost

                # you can implement functionality to compute the improvement in-between, these code block
                self._cur_sol = self._min_sol
                self._cur_cost = self._cur_cost

                self._cur_req_idx += 1
                self.update_planning_hour()

        self._cur_sol.assign_runs()
        self._cur_cost = self._min_sol.compute_cost()
        self._min_sol = self._cur_sol
        self._min_cost = self._cur_cost

        # final simulated annealing search
        _, _, min_sol, min_cost, summary_file = self._run_with_anytime(self._cur_sol)

        self._min_sol = min_sol
        self._min_cost = min_cost

        # optional plots for objective cost, runs, and vehicles
        if summary_file is not None:
            summary_file.plot("iteration", "objective_cost")
            summary_file.plot("iteration", "runs", "run")
            summary_file.plot("iteration", "vehicles", "veh")

        # compute the improvement at the end of the day for operation cost
        improve_percentage = round(100.0 * (self._cur_cost - self._min_cost) / self._cur_cost, 3)
        logger.info(f"Improvement in operational cost {improve_percentage}%")

        self._min_sol.verify()
        if write_summary and not self._config.skip_dump_process:
            self._min_sol.write_solution()
            self._min_sol.write_summary()
            create_dir(f"{RESULT_DIR}/{self._args.agency}/")
            optimal_time_window = open(f"{RESULT_DIR}/{self._args.agency}/{self._args.date}.csv", "w+")
            optimal_time_window.write("booking_id,p_earl_t\n")
            for req in self._cur_sol.requests:
                optimal_time_window.write(f"{req.booking_id},{req.pick_up_node.earliest_arrival}\n")
            optimal_time_window.close()

    def _run_with_anytime(self, cur_sol, i=-1):
        # solve the greedy version of the problem
        cur_sol.solve_limited(self._cur_req_idx)

        cur_cost = cur_sol.compute_cost()

        # run the simulated annealing search for the given time
        suffix = ""
        if i != -1:
            suffix = f"r_{i}"
        min_sol, min_cost, summary_file = sim_anneal_search(
            self, self._args, cur_sol, self.anytime_plan_duration, write_summary=False, special_suffix=suffix
        )
        return cur_sol, cur_cost, min_sol, min_cost, summary_file

    def get_min_solution(self):
        """
        :return: the minimum solution
        """
        return self._min_sol

    def get_cost(self):
        """
        :return: the minimum cost
        """
        return self._min_cost

    def get_solution_details(self):
        return self._min_sol.get_solution_details()


class BaselineBasedOnlineBookingSimulator(OnlineBookingSimulator):
    """
        Implementation of online-booking simulator that allows to run the custom
        anytime algorithm
    """

    def _create_custom_anytime_solver(self):
        """
            this will create custom anytime solver
        """
        custom_anytime_solver = self._custom_anytime_solver_cls(
            run_cls=self._run_cls,
            sample_inst_config=self._sample_inst_config,
            config=self._config,
            custom_args=self._args
        )
        return custom_anytime_solver

    def _run_with_anytime(self, cur_sol, i=-1):
        # solve the wrapper version of the problem
        if hasattr(cur_sol, 'set_configs'):
            cur_sol.set_configs(search_duration=self.anytime_plan_duration)
        cur_sol.solve_limited(self._cur_req_idx)
        cur_cost = cur_sol.compute_cost()
        return cur_sol, cur_cost, cur_sol, cur_cost, None
