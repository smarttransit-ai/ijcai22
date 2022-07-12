########################################
#            RUN OPTIMIZER             #
########################################

import sys
import warnings
from datetime import datetime

from common.parsers.ArgParser import CustomArgParser
from common.util.common_util import logger

warnings.filterwarnings('ignore')


def parameter_check(input_args):
    start_prob = float(input_args.start_prob)
    end_prob = float(input_args.end_prob)
    alter_prob = float(input_args.alter_prob)
    duration = int(input_args.plan_duration)

    alter_condition = 0 < alter_prob < 1
    start_end_condition = 0 < end_prob < start_prob < 1
    duration_condition = duration > 0

    failed = False
    if not alter_condition:
        logger.error("Altering probability should be greater than 0 and less than 1")
        failed = True

    if not start_end_condition:
        logger.error("Start probability should be greater than End probability")
        failed = True

    if not duration_condition:
        logger.error("Plan duration should be greater than 0")
        failed = True

    if failed:
        logger.warning("Optimizer exiting...")
        sys.exit(-1)


def get_solver(input_args):
    if input_args.algo == "agent_without_anytime":
        from base.generator import create_online_booking_simulation
        parameter_check(input_args)
        solver = create_online_booking_simulation(custom_args=input_args)
        solver.only_real_time = True

    elif input_args.algo == "agent_with_anytime":
        from base.generator import create_online_booking_simulation
        parameter_check(input_args)
        solver = create_online_booking_simulation(custom_args=input_args)
        solver.only_real_time = False

    elif input_args.algo == "greedy":
        from base.generator import create_greedy
        solver = create_greedy(custom_args=input_args)

    elif input_args.algo == "sim_anneal":
        from base.generator import create_sim_anneal
        parameter_check(input_args)
        solver = create_sim_anneal(custom_args=input_args)

    elif input_args.algo.endswith("routing"):
        from base.generator import create_routing, create_routing_online_booking_simulator
        if input_args.algo == "routing":
            solver = create_routing(custom_args=input_args)

        elif input_args.algo == "agent_with_anytime_routing":
            if input_args.anytime_algo != "default":
                solver = create_routing_online_booking_simulator(custom_args=input_args)
                solver.prefix = f"agent_with_anytime_{input_args.anytime_algo}"
            else:
                solver = create_routing(custom_args=input_args)
                solver.use_custom_tight_window = True
                solver.trainer = "agent_with_anytime"

        elif input_args.algo == "agent_without_anytime_routing":
            if input_args.anytime_algo == "default":
                solver = create_routing(custom_args=input_args)
                solver.use_custom_tight_window = True
                solver.trainer = "agent_without_anytime"
            else:
                raise ValueError(f"The routing can't be run with {input_args.algo} in online booking simulation")
        else:
            raise ValueError(f"The routing wrapper {input_args.algo} doesn't implemented")

    elif input_args.algo.endswith("vroom"):
        from base.generator import create_vroom, create_vroom_online_booking_simulation
        if input_args.algo == "vroom":
            solver = create_vroom(custom_args=input_args)

        elif input_args.algo == "agent_with_anytime_vroom":
            if input_args.anytime_algo != "default":
                solver = create_vroom_online_booking_simulation(custom_args=input_args)
                solver.prefix = f"agent_with_anytime_{input_args.anytime_algo}"
            else:
                solver = create_vroom(custom_args=input_args)
                solver.use_custom_tight_window = True
                solver.trainer = "agent_with_anytime"

        elif input_args.algo == "agent_without_anytime_vroom":
            if input_args.anytime_algo == "default":
                solver = create_vroom(custom_args=input_args)
                solver.use_custom_tight_window = True
                solver.trainer = "agent_without_anytime"
            else:
                raise ValueError(f"The vroom can't be run with {input_args.algo} in online booking simulation")
        else:
            raise ValueError(f"The vroom wrapper {input_args.algo} doesn't implemented")
    else:
        raise ValueError(f"The heuristic algorithm {input_args.algo} doesn't implemented")
    return solver


def run_optimizer(_args):
    solver = get_solver(_args)
    if hasattr(solver, "solve"):
        solver.solve(write_summary=True)
    else:
        solver.search(write_summary=True)
    return solver


if __name__ == '__main__':
    start_time = datetime.now()
    arg_parser = CustomArgParser()
    args = arg_parser.parse_args()
    run_optimizer(args)
    logger.info(f"Time taken for the computation: {(datetime.now() - start_time).total_seconds()}")
