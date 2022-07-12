from enum import Enum

from base.generator import create_online_booking_simulation, create_routing_online_booking_simulator, \
    create_vroom_online_booking_simulation
from common.constant.constants import get_time_matrix_src
from common.util.common_util import logger


class AnyTimeAlgo(Enum):
    DEFAULT = "default"
    VROOM = "vroom"
    ROUTING = "routing"


class RealTimeEnvBase(object):
    def __init__(self, args):
        anytime_algo = args.anytime_algo
        self.anytime_algo = AnyTimeAlgo.DEFAULT
        self.solver = self.get_anytime_solver(args, anytime_algo)
        self.state = self.solver.get_state()
        self.solver.only_real_time = True
        self.prefix = "agent_with_anytime"

    def get_anytime_solver(self, args, algo_name):
        time_matrix_src = get_time_matrix_src(args.time_matrix_src)
        if algo_name == AnyTimeAlgo.DEFAULT.value:
            solver = create_online_booking_simulation(custom_args=args, travel_time_data_set=time_matrix_src)
            self.anytime_algo = AnyTimeAlgo.DEFAULT
        elif algo_name == AnyTimeAlgo.VROOM.value:
            solver = create_vroom_online_booking_simulation(custom_args=args, travel_time_data_set=time_matrix_src)
            self.anytime_algo = AnyTimeAlgo.VROOM
        elif algo_name == AnyTimeAlgo.ROUTING.value:
            solver = create_routing_online_booking_simulator(custom_args=args, travel_time_data_set=time_matrix_src)
            self.anytime_algo = AnyTimeAlgo.ROUTING

        else:
            raise ValueError(f"invalid anytime algorithm : {algo_name}")
        return solver

    def reset(self):
        self.solver.reset()
        self.state = self.solver.get_state()
        return self.state

    def step(self, action):
        raise NotImplementedError


class RealTimeEnv(RealTimeEnvBase):
    def __init__(self, args):
        super(RealTimeEnv, self).__init__(args)
        self.prefix = "agent_without_anytime"
        # this is a bad practice, later I will fix this
        if self.anytime_algo != AnyTimeAlgo.DEFAULT:
            self.prefix += f"_{self.anytime_algo.value}"
        logger.info("Setting RL-Environment (Without Anytime)")

    def step(self, action):
        done, solution, cost = self.solver.optimal_time_window_rl(action)
        reward = -1 * cost
        next_state = self.solver.get_state()
        return reward, next_state, done


class RealTimeAnyTimeEnv(RealTimeEnvBase):
    def __init__(self, args):
        super(RealTimeAnyTimeEnv, self).__init__(args)
        self.prefix = "agent_with_anytime"
        if self.anytime_algo != AnyTimeAlgo.DEFAULT:
            self.prefix += f"_{self.anytime_algo.value}"
        logger.info(f"Setting RL-Environment (With Anytime) Anytime algo: {self.anytime_algo.value}")

    def step(self, action):
        done, solution, cost = self.solver.optimal_time_window_rl_with_sa(action)
        reward = -1 * cost
        next_state = self.solver.get_state()
        return reward, next_state, done
