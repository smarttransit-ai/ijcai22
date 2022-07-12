import logging
import multiprocessing
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
from pandas import DataFrame

from common.constant.constants import COMMON_DIR, BASE_DATA_DIR
from common.parsers.ConfigParser import AgencyConfigParser


def file_exists(file_name, extension):
    """
    :param file_name: file name with full path
    :param extension: file extension
    :return: returns whether file exists or not
    """
    exists = False
    if os.path.exists(file_name + extension):
        exists = True
    return exists


def directory_exists(dir_name):
    """
    :param dir_name: directory name with full path
    :return: returns whether directory exists or not
    """
    exists = False
    if file_exists(dir_name, ""):
        exists = True
    return exists


def create_dir(dir_name):
    """
    :param dir_name: name of directory, which need to be created
    """
    if "." in dir_name:
        last_slash = dir_name.rfind("/")
        if last_slash != -1:
            dir_name = dir_name[:last_slash + 1]
    if not directory_exists(dir_name):
        if "/" not in dir_name and "." not in dir_name:
            try:
                os.mkdir(dir_name)
            except FileExistsError:
                print(dir_name + " already exists")
        else:
            try:
                os.makedirs(dir_name)
            except FileExistsError:
                print(dir_name + " already exists")


def delete_dir(dir_name):
    """
    :param dir_name: name of directory, which need to be deleted
    """
    if directory_exists(dir_name):
        shutil.rmtree(dir_name)


def delete_file(file_name, extension):
    """
    :param file_name: name of file, which need to be deleted
    :param extension: file extension
    """
    if file_exists(file_name, extension):
        os.remove(file_name + extension)


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s"
    date_fmt = '%Y-%m-%d %H:%M:%S'

    FORMATS = {
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.date_fmt)
        return formatter.format(record)


class LogFormatter(logging.Formatter):
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    date_fmt = '%Y-%m-%d %H:%M:%S'

    def format(self, record):
        formatter = logging.Formatter(self.fmt, datefmt=self.date_fmt)
        return formatter.format(record)


logger = logging.getLogger("ParaTransit")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
create_dir("log")
file_handler = logging.FileHandler(f'log/pt_optimizer_{datetime.now().timestamp()}.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(LogFormatter())
logger.addHandler(file_handler)


def extract(main_dir, file_ending=""):
    """
    :param main_dir: name of the main directory
    :param file_ending: the ending of the file
    :return: returns the list of files with corresponding file ending in the specified main directory
    """
    _files = []
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file.endswith(file_ending):
                __file_path = os.path.join(root, file)
                _files.append(__file_path)
    return _files


def convert_sec_to_hh_mm_ss(seconds):
    """
    :param seconds: time in seconds
    :return: convert the seconds to the format hh:mm:ss
    """
    _min, _sec = divmod(seconds, 60)
    _hour, _min = divmod(_min, 60)
    return "%d:%02d:%02d" % (_hour, _min, _sec)


def format_time(time_in_sec):
    """
    :param time_in_sec: provide input time in seconds
    :return: formatted time in HH:MM:SS
    """
    return time.strftime('%H:%M:%S', time.gmtime(time_in_sec))


def get_time_stamp(date_string, time_string=""):
    """
    :param date_string: date in string format
    :param time_string: time in string format
    :return: timestamp
    """
    if time_string == "":
        date = datetime.strptime(date_string, "%Y-%m-%d")
    else:
        date = datetime.strptime(time_string + " " + date_string, "%H:%M:%S %Y-%m-%d")
    time_tuple = date.timetuple()
    return time.mktime(time_tuple)


def del_n_create_dir(dir_name):
    """
    :param dir_name: name of directory, which need to be first deleted and then created again
    """
    delete_dir(dir_name)
    create_dir(dir_name)


def soft_max(x):
    # for large negative number np.exp will provide zero, to overcome this
    # I have made this fix
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_agency_config(selected_agency, custom_config_file=None):
    """
    :param selected_agency: selected agency
    :param custom_config_file: custom configuration file
    :return: agency configuration
    """
    if custom_config_file is None:
        custom_config_file = f"{COMMON_DIR}/{selected_agency}.ini"
    AgencyConfigParser.instance().config(custom_config_file)
    return AgencyConfigParser.instance()


def read_data_file(
        selected_agency="AGENCY_A",
        selected_date=0,
        custom_file=None,
        date_filter_off=False,
        only_main_authority=True,
        ignore_wc_runs=False
):
    """
    :param selected_agency: agency used to filter the para-transit trips
    :param selected_date: date used to filter the para-transit trips
    :param custom_file: provide custom file to read data
    :param date_filter_off: turning off the date filter to read entire set of trips
    :param only_main_authority: by tuning on, filter the data for main authority
    :param ignore_wc_runs: this enables filtering WC runs
    :return: returns the dataframe of selected data
    """
    import pandas as pd
    columns_maps = {
        'bookingid': 'Booking Id', 'ldate': 'Date',
        'Pickup LAT': 'Pickup lat', 'Pickup LON': 'Pickup lon',
        'Dropoff LAT': 'Dropoff lat', 'Dropoff LON': 'Dropoff lon',
    }
    expected_cols = [
        'Sch Time in HH:MM:SS', 'Date', 'Pickup lat',
        'Pickup lon', 'Dropoff lat', 'Dropoff lon', 'Passenger Types'
    ]

    if custom_file is None:
        df = DataFrame()
        base_data_dir = BASE_DATA_DIR.format(selected_agency)
        files = extract(base_data_dir, ".csv")
        for file_path in files:
            if "trips" in file_path:
                df_sub = pd.read_csv(file_path, low_memory=False)
                df = df.append(df_sub, ignore_index=True)
    else:
        df = pd.read_csv(custom_file)

    df = df.rename(columns=columns_maps)
    if "Booking Id" in df.columns:
        expected_cols = ["Booking Id"] + expected_cols
    df = df[expected_cols]
    df = df.drop_duplicates(keep='first')

    # # remove rows which has missing passenger details
    # # i.e., passenger types and number of passenger for each type
    # df = df.dropna(subset=["Passenger Types"])
    # filter out non-agency trips
    if only_main_authority:
        if "Provider" in df.columns:
            df = df[df["Provider"] == get_agency_config(selected_agency).main_transit_authority]
    # filter-out WC runs
    if ignore_wc_runs:
        run_values = set(df["Run"].to_list())
        for run_value in run_values:
            if "WC" in run_value:
                df = df[df["Run"] != run_value]
    if len(df) == 0:
        logger.error('no data found')
        sys.exit(-1)
    if not date_filter_off:
        df = df[df["Date"] == selected_date]
        if len(df) == 0:
            logger.error(f'no data found for the selected date {selected_date}')
            sys.exit(-1)
        df = df.sort_values(by="Booking Id")
    return df


def visualize(pt_obj, depot_coordinates, limits):
    """
    :param pt_obj: optimizer object
    :param depot_coordinates: depot coordinates for the system
    :param limits: x-axis and y-axis limits
    """
    import math
    import random
    import matplotlib.pyplot as plt
    circle_rad = 0.005
    inner_circle_rad = 0.004
    fig, ax = plt.subplots()
    plt.xlim(limits["x"])
    plt.ylim(limits["y"])
    for run_id in pt_obj.run_plans.keys():
        i = list(pt_obj.run_plans.keys()).index(run_id)
        random.seed(i + 1000)
        r = lambda: random.randint(0, 255)
        v_color = '#%02X%02X%02X' % (r(), r(), r())
        full_plan = pt_obj.run_plans[run_id]
        if len(full_plan) > 1:
            for assign_data in full_plan:
                diff_x = assign_data.e_cor[0] - assign_data.s_cor[0]
                diff_y = assign_data.e_cor[1] - assign_data.s_cor[1]
                dist = math.sqrt(math.pow(diff_x, 2) + math.pow(diff_y, 2))
                distance_diff = dist - 2 * circle_rad
                if dist == 0:
                    dist = 1
                radius_dist_ratio = circle_rad / dist
                new_diff_x = radius_dist_ratio * diff_x
                new_diff_y = radius_dist_ratio * diff_y
                diff_o_x = diff_x * distance_diff / dist
                diff_o_y = diff_y * distance_diff / dist
                cir = plt.Circle(assign_data.s_cor, circle_rad, color=v_color)
                ax.add_artist(cir)
                cir = plt.Circle(assign_data.s_cor, inner_circle_rad, color="white")
                ax.add_artist(cir)
                txt_idx = plt.Text(assign_data.s_cor[0] - 0.002, assign_data.s_cor[1] - 0.002, f"{assign_data.s_idx}",
                                   fontproperties={"size": 4})
                ax.add_artist(txt_idx)
                arr = plt.Arrow(assign_data.s_cor[0] + new_diff_x, assign_data.s_cor[1] + new_diff_y,
                                diff_o_x, diff_o_y, width=0.005, color=v_color)
                ax.add_artist(arr)
                cir = plt.Circle(assign_data.e_cor, circle_rad, color=v_color)
                ax.add_artist(cir)

    cir = plt.Circle(depot_coordinates, circle_rad, color="black")
    ax.add_artist(cir)
    cir = plt.Circle(depot_coordinates, inner_circle_rad, color="white")
    ax.add_artist(cir)
    create_dir(pt_obj.summary_file_jpg)
    fig.savefig(pt_obj.summary_file_jpg, dpi=400)
    plt.close()


class PyParallelWrapper(object):
    """
        Wrapper class for parallel processing options
    """

    def __init__(self, _function, _arguments, _no_of_workers, _wait_for_response=False, _skip_logging=True):
        self._function = _function
        self._arguments = _arguments
        self._no_of_workers = min(len(_arguments), multiprocessing.cpu_count() - 1, _no_of_workers)
        self._wait_for_responses = _wait_for_response

    def _execute_single_proc(self):
        """
            special case when it was assigned to run in single process
        """
        logger.warning("Multiprocessing called with only one processor")
        responses = []
        for _args in self._arguments:
            response = self._function(_args)
            responses.append(response)
        return responses


def multi_processing_wrapper(function, arguments, no_of_workers, wait_for_response=False, skip_logging=True):
    """
    This is a Wrapper function that can run the instances in the multiprocessing mode and return results
    :param function: function to evaluate
    :param arguments: contains the list of arguments
    :param no_of_workers: provide the number of workers
    :param wait_for_response: by enabling this wrapper function wait-for process completion
    :param skip_logging: skip logging details of multiprocessing
    :return: return the results at the end
    """

    class MPWrapper(PyParallelWrapper):
        """
            Wrapper Class to Multiprocessing
        """

        def __init__(self, _function, _arguments, _no_of_workers, _wait_for_response=False, _skip_logging=True):
            super(MPWrapper, self).__init__(_function, _arguments, _no_of_workers, _wait_for_response, _skip_logging)
            if not _skip_logging:
                logger.info(f"Running multiprocessing mode, no of processes {self._no_of_workers}")
            if multiprocessing.get_start_method() != 'fork':
                multiprocessing.set_start_method('fork', force=True)

        def execute(self):
            """
            :return: run multi-processing
            """
            if self._no_of_workers > 1:
                p = multiprocessing.Pool(self._no_of_workers)
                responses = p.map(self._function, self._arguments)
                if self._wait_for_responses:
                    for _ in responses:
                        pass
            elif self._no_of_workers == 1:
                responses = self._execute_single_proc()
            else:
                raise ValueError("Number of workers should be greater than 1/0")
            return responses

    mp_wrapper = MPWrapper(function, arguments, no_of_workers, wait_for_response, skip_logging)
    return mp_wrapper.execute()


def ray_wrapper(function, arguments, no_of_workers, wait_for_response=False, skip_logging=True):
    """
    This is a Wrapper function that can run the instances in using ray and return results
    :param function: function to evaluate
    :param arguments: contains the list of arguments
    :param no_of_workers: provide the number of workers
    :param wait_for_response: by enabling this wrapper function wait-for process completion
    :param skip_logging: skip logging details of ray
    :return: return the results at the end
    """
    import ray

    class RayWrapper(PyParallelWrapper):
        """
            Wrapper Class to Ray
        """

        def __init__(self, _function, _arguments, _no_of_workers, _wait_for_response=False, _skip_logging=True):
            super(RayWrapper, self).__init__(_function, _arguments, _no_of_workers, _wait_for_response, _skip_logging)
            if not _skip_logging:
                logger.info(f"Running using ray, no of cpus: {self._no_of_workers}")
            ray.init(num_cpus=self._no_of_workers)

        def execute(self):
            """
                Run ray-based multiple process
            """

            @ray.remote
            def remote_func(args):
                return self._function(args)

            if self._no_of_workers > 1:
                result_ids = [remote_func.remote(_args) for _args in self._arguments]
                responses = ray.get(result_ids)

                if self._wait_for_responses:
                    for _ in responses:
                        pass
            elif self._no_of_workers == 1:
                responses = self._execute_single_proc()
            else:
                raise ValueError("Number of workers should be greater than 1/0")
            return responses

    r_wrapper = RayWrapper(function, arguments, no_of_workers, wait_for_response, skip_logging)
    return r_wrapper.execute()
