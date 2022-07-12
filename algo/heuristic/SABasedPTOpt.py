##################################################################
#          SIMULATED ANNEALING ALGORITHM IMPLEMENTATION          #
##################################################################
#  This is simulated annealing algorithm implementation to solve #
#  the custom VRP problem with time windows and route-length     #
#  limitations                                                   #
##################################################################

from base.common.util import sim_anneal_search
from common.constant.constants import AGENCY_ITER_DIR
from common.util.common_util import logger


class SABasedPTOpt(object):
    """
        This class stores custom solver setup to solve para-transit optimization problem,
        using Simulated Annealing
    """

    def __init__(
            self,
            run_cls,
            greedy_cls,
            sample_inst_config=None,
            config=None,
            custom_args=None,
            custom_agency_config=None
    ):
        self.__run_cls = run_cls
        self.__greedy_cls = greedy_cls
        self.__sample_inst_config = sample_inst_config
        self.__config = config
        self.__args = custom_args
        self.__agency_config = custom_agency_config
        summary_file_common = f"sa_t{self.__sample_inst_config.no_of_trips}_" + \
                              f"r{self.__sample_inst_config.no_of_runs}_" + \
                              f"v{self.__sample_inst_config.no_of_vehicles}_" + \
                              f"{self.__sample_inst_config.random_seed}"
        self.summary_iter_file_name = f"{AGENCY_ITER_DIR}/{summary_file_common}"
        self.summary_iter_file_name = self.summary_iter_file_name.format(
            self.__sample_inst_config.selected_agency, self.__sample_inst_config.selected_date
        )
        self.__min_sol = None
        self.__min_cost = None

    def search(self, write_summary=False):
        """
            search for optimal solutions
        """
        greedy = self.__greedy_cls(
            prefix="sim_anneal",
            run_cls=self.__run_cls,
            sample_inst_config=self.__sample_inst_config,
            config=self.__config,
            custom_args=self.__args
        )
        greedy.solve()
        initial_cost = greedy.compute_cost()
        duration = int(self.__args.plan_duration)
        min_sol, min_cost, summary_file = sim_anneal_search(
            self, self.__args, greedy, duration=duration, write_summary=True
        )
        self.__min_sol = min_sol
        self.__min_cost = min_cost
        self.__min_sol.verify()
        if summary_file is not None:
            summary_file.plot("iteration", "objective_cost")
            summary_file.plot("iteration", "runs", "run")
            summary_file.plot("iteration", "vehicles", "veh")
        improve_percentage = round(100.0 * (initial_cost - min_cost) / initial_cost, 3)
        logger.info(f"Improvement in operational cost {improve_percentage}%")
        if write_summary and not self.__config.skip_dump_process:
            self.__min_sol.write_solution()
            self.__min_sol.write_summary()

    def get_min_solution(self):
        """
        :return: the minimum solution
        """
        return self.__min_sol

    def get_cost(self):
        """
        :return: the minimum cost
        """
        return self.__min_cost

    def get_solution_details(self):
        return self.__min_sol.get_solution_details()
