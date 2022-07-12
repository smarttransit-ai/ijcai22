import os
from enum import Enum

MIN_SEC_SWAP = 60
DAY_IN_MINUTES = 1440
ADDITIONAL_TIME = 1440
VEHICLE_TRAVEL_MAX_TIME = (DAY_IN_MINUTES + ADDITIONAL_TIME) * MIN_SEC_SWAP

# data collection related constants
DATA_FETCH_PORT = 9000
DATA_FETCH_DATE = "2021-2-28"
DATA_FETCH_TIME = "12:00"

REQUEST_URL_FRMT = "http://localhost:{}/otp/routers/default/plan?fromPlace={},{}&toPlace={},{}&time={}&date={}" \
                   "&mode=CAR&maxWalkDistance=0&arriveBy=false&wheelchair=false&debugItineraryFilter=false&" \
                   "locale=en&itinIndex=0"

LOCAL_DIR = os.getcwd()
CACHE_DIR = f"{LOCAL_DIR}/cache"

COMMON_DIR = f"{LOCAL_DIR}/common"

# DATA SECTION
DATA_DIR = f"{LOCAL_DIR}/data"
AGENCY_DATA_DIR = f"{DATA_DIR}" + "/{}"
AGENCY_TRAVEL_TIMES_DIR = f"{DATA_DIR}" + "/{}/travel_times"
BASE_DATA_DIR = f"{AGENCY_DATA_DIR}/base"
BASE_DATA_FILE = BASE_DATA_DIR + "/para_transit_trips_{}"

# file extensions
INDEX_EXTENSION = ".idx"
PICKLE_EXTENSION = ".pickle"
PICKLE_SHORT_EXTENSION = ".pkl"
NUMPY_EXTENSION = ".npy"
CSV_EXTENSION = ".csv"
SH_EXTENSION = ".sh"
PY_EXTENSION = ".py"

# file_name for input data
FULL_DATA_DIR = f"{AGENCY_DATA_DIR}/full_data"
DAY_DATA_DIR = f"{AGENCY_DATA_DIR}/day_base"
STATS_DATA_DIR = f"{AGENCY_DATA_DIR}/stats"
MODELS_DATA_DIR = f"{AGENCY_DATA_DIR}/models"
TRAVEL_MATRICES_DATA_DIR = f"{AGENCY_DATA_DIR}/matrices"

LOCATIONS_FILE_NAME = f"{TRAVEL_MATRICES_DATA_DIR}/locations"

# travel times of locations
LOCATIONS_DATE_FILE_NAME = DAY_DATA_DIR + "/{}/locations"
TRAVEL_TIMES_DATE_FILE_NAME = DAY_DATA_DIR + "/{}/{}/travel_times"
TRAVEL_DISTANCES_DATE_FILE_NAME = DAY_DATA_DIR + "/{}/{}/travel_distances"
MISSED_LOCATIONS_FILE_NAME = DAY_DATA_DIR + "/{}/{}/missed_locations"

# travel time matrix sources
NXN_STR = "nxn"


class TimeMatrixSource(Enum):
    NXN = NXN_STR


def get_time_matrix_src(time_matrix_src_str):
    if time_matrix_src_str.lower() == NXN_STR:
        return TimeMatrixSource.NXN


# RESULT SECTION
RESULT_BASE_DIR = f"{LOCAL_DIR}/result"
COUNTER = 0
while os.path.exists(f"{RESULT_BASE_DIR}-{COUNTER}"):
    COUNTER += 1
RESULT_DIR = f"{RESULT_BASE_DIR}-{COUNTER}"
AGENCY_RESULT_DIR = f"{RESULT_DIR}/" + "{}"
AGENCY_ASSIGNS_DIR = f"{AGENCY_RESULT_DIR}/" + "{}/assigns"
AGENCY_DUMP_DIR = f"{AGENCY_RESULT_DIR}/" + "{}/dump"
AGENCY_IMAGES_DIR = f"{AGENCY_RESULT_DIR}/" + "{}/images"
AGENCY_PLANS_DIR = f"{AGENCY_RESULT_DIR}/" + "{}/plans"
AGENCY_ITER_DIR = f"{AGENCY_RESULT_DIR}/" + "{}/iterations"

# dumps of input data and processed data as python byte object
GENERIC_DATA_FILE_NAME = AGENCY_DUMP_DIR + "/generic_data_{}_{}_{}"
ALGO_DATA_FILE_NAME = AGENCY_DUMP_DIR + "/algo_data_{}_{}_{}_{}_{}"

# dumps of input data, and processed data in a human-readable format
GENERIC_DATA_DIR = AGENCY_DUMP_DIR + "/generic_data_{}_{}_{}_{}"
SHORTEN_DATA_FILE_NAME = f"{GENERIC_DATA_DIR}/shorten_data.csv"
EXPANDED_DATA_FILE_NAME = f"{GENERIC_DATA_DIR}/expanded_data.csv"
TIME_MATRIX_FILE_NAME = f"{GENERIC_DATA_DIR}/time_matrix.csv"
VEHICLE_INFO_FILE_NAME = f"{GENERIC_DATA_DIR}/vehicle_info.csv"
