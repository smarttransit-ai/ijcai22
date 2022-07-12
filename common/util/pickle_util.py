import pickle

import numpy as np

from common.constant.constants import PICKLE_EXTENSION, PICKLE_SHORT_EXTENSION, NUMPY_EXTENSION
from common.util.common_util import create_dir, file_exists


def dump_obj(obj, file_name, args=None):
    """
    :param obj: python object
    :param file_name: file to save the python object
    :param args: custom arguments to change the file
    """
    if args is not None:
        file_name = file_name.format(*args)
    if not (file_name.endswith(PICKLE_EXTENSION) or file_name.endswith(PICKLE_SHORT_EXTENSION)):
        file_name += PICKLE_EXTENSION
    create_dir(file_name)
    dump_file = open(file_name, "wb")
    pickle.dump(obj, dump_file)
    dump_file.close()


def load_obj(file_name, args=None):
    """
    :param file_name: file to load the python object
    :param args: custom arguments to change the file
    :return: python object
    """
    if args is not None:
        file_name = file_name.format(*args)
    if not (file_name.endswith(PICKLE_EXTENSION) or file_name.endswith(PICKLE_SHORT_EXTENSION)):
        file_name += PICKLE_EXTENSION
    dump_file = open(file_name, "rb")
    obj = pickle.load(dump_file)
    dump_file.close()
    return obj


def load_np_obj(file_name, args=None):
    """
    :param file_name: file to load the python object
    :param args: custom arguments to change the file
    :return: python object
    """
    if args is not None:
        file_name = file_name.format(*args)
    if not file_name.endswith(NUMPY_EXTENSION):
        file_name += NUMPY_EXTENSION
    with open(file_name, "rb") as dump_file:
        obj = np.load(dump_file)
    return obj


def dump_exists(file_name):
    """
    :param file_name: file name with full path
    :return: returns whether dump exists or not
    """
    return file_exists(file_name, PICKLE_EXTENSION)
