import math
import random
from datetime import datetime

from common.constant.constants import CSV_EXTENSION
from common.util.common_util import create_dir, logger
from common.writer.FileWriter import FileWriter


def nearest_neighbour(solution, alter_prob, random_seed, k, verify=False):
    """
    :param solution: current solution
    :param alter_prob: alteration probability
    :param random_seed: random seed
    :param k: iteration
    :param verify: verify whether all the requests present after the mutation operations
    :return: return the swapped solution
    """
    temp_solution = solution.copy()
    no_of_alterations = max(1, int(alter_prob * len(temp_solution.get_active_runs())))
    logger.info(f"Number of alterations: {no_of_alterations}")
    for i in range(no_of_alterations):
        random.seed(random_seed + i + k)
        choice = random.choice([0, 1])
        if choice == 0:
            random.seed(random_seed + i + k + choice)
            split_choice = random.choice([2, 3, 4])
            pattern_choice = random.choice([True])
            temp_solution = temp_solution.split_and_merge_runs(
                n=split_choice, fixed_pattern=pattern_choice, verify=verify
            )
        elif choice == 1:
            temp_solution = temp_solution.swap(verify=verify)
    return temp_solution


def sim_anneal_search(cls, args, min_sol, duration=math.inf, write_summary=False, special_suffix=""):
    """
    :param cls: Class object to solve the simulated annealing problem
    :param args: system arguments
    :param min_sol: initial solution
    :param duration: duration to run the algorithm
    :param write_summary: enabling this will write summary
    :param special_suffix: suffix for summary file
    :return: do the simulated annealing search for the minimum solution,
    return the minimum solution, costs and summary file
    """
    start_prob = float(args.start_prob)
    end_prob = float(args.end_prob)
    alter_prob = float(args.alter_prob)

    logger.info(
        f"Simulated annealing configs: running time: {duration}, start prob: {start_prob}, " +
        f"end prob: {end_prob}, swap prob: {alter_prob}"
    )

    current_sol = min_sol.copy()
    current_cost = min_sol.compute_cost()
    min_cost = current_cost

    temp_start = -1.0 / math.log(start_prob)
    temp_end = -1.0 / math.log(end_prob)
    rate_of_temp = (temp_end / temp_start) ** (1.0 / (duration - 1.0))

    selected_temp = temp_start
    delta_e_avg = 0.0
    number_of_accepted = 1
    summary_file_name = cls.summary_iter_file_name + f"s_{start_prob}_e_{end_prob}_a_{alter_prob}"
    if special_suffix != "":
        summary_file_name += special_suffix
    create_dir(summary_file_name + CSV_EXTENSION)
    summary_file = None

    if write_summary:
        # added this to handle issues when using it on clusters
        summary_file = FileWriter(summary_file_name, CSV_EXTENSION)
        summary_file.write("iteration,objective_cost,runs,vehicles,requests")
        summary_file.write(
            [0, current_cost, current_sol.no_of_runs, current_sol.no_of_vehicles, current_sol.no_of_requests]
        )
    i = 0
    start = datetime.now()
    while (datetime.now() - start).total_seconds() < duration:
        nn_sol = nearest_neighbour(current_sol, alter_prob, int(args.random_seed), i, verify=False)
        nn_cost = nn_sol.compute_cost()
        logger.info(f"Cycle: {i + 1} with Temperature: {selected_temp} and Cost: {nn_cost}")
        delta_e = abs(nn_cost - current_cost)
        if nn_cost > current_cost:
            if i == 0:
                delta_e_avg = delta_e
            denominator = (delta_e_avg * selected_temp)
            p = math.exp(-1 * math.inf) if denominator == 0 else math.exp(-delta_e / denominator)
            accept = True if random.random() < p else False
        else:
            accept = True
        # save current minimum to avoid losing details due to crash
        if min_cost > nn_cost:
            min_sol = nn_sol
            min_cost = nn_cost

        if accept:
            current_sol = nn_sol
            current_cost = nn_cost
            delta_e_avg = delta_e_avg + (delta_e - delta_e_avg) / number_of_accepted
            number_of_accepted += 1
        selected_temp = temp_start * math.pow(rate_of_temp, duration)
        if summary_file is not None:
            summary_file.write([i + 1, nn_cost, nn_sol.no_of_runs, nn_sol.no_of_vehicles, nn_sol.no_of_requests])
        i += 1
    if summary_file is not None:
        summary_file.close()
    return min_sol, min_cost, summary_file
