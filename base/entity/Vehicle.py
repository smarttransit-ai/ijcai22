from base.common.MappedList import MappedList
from base.data.GenDataLoader import GenDataLoader


class Vehicle(object):
    """
        This class stores the details of Vehicle
    """

    def __init__(self, name):
        self.__name = name
        self.__runs = MappedList()

    def copy(self):
        veh_cpy = Vehicle(self.__name)
        veh_cpy.__runs = MappedList(self.__runs.get_list())
        return veh_cpy

    def get_name(self):
        """
        :return: the name of the vehicle
        """
        return self.__name

    def get_runs(self):
        """
        :return: the _runs of the vehicle
        """
        return self.__runs.get_list()

    def check_assign(self, run):
        """
        :param run: run that needs to assign to this vehicle
        :return: whether the run can be assigned to vehicle or not
        """
        return self.assign(run, check_only=True)

    def assign(self, run, check_only=False):
        """
        :param run: run that needs to assign to this vehicle
        :param check_only: check only whether the run can be assigned to vehicle or not
        :return: whether the run is assigned to this vehicle or not
        """
        success = False
        temp_runs = []
        if self.__runs.size() == 0:
            temp_runs = self.__runs.get_list() + [run]
            success = True
        else:
            buffer = GenDataLoader.instance().agency_config.buffer_time
            for i, ex_run in enumerate(self.__runs.get_list()):
                if ex_run.get_end() + buffer <= run.get_start():
                    if i == len(self.__runs.get_list()) - 1:
                        temp_runs = self.__runs.get_list() + [run]
                        success = True
                    else:
                        if run.get_end() + buffer <= self.__runs.get_list()[i + 1].get_start():
                            temp_runs = self.__runs.get_list()[:i + 1] + [run] + self.__runs.get_list()[i + 1:]
                            success = True
                if run.get_end() + buffer <= ex_run.get_start():
                    if i == 0:
                        temp_runs = [run] + self.__runs.get_list()
                        success = True
                    else:
                        if self.__runs.get_list()[i - 1].get_end() + buffer <= run.get_start():
                            temp_runs = self.__runs.get_list()[:i] + [run] + self.__runs.get_list()[i:]
                            success = True
                if success:
                    break

        if success and not check_only:
            self.__runs = MappedList(temp_runs.copy())
        return success

    def remove(self, run):
        """
        :param run: run that needs to be removed
        """
        if run in self.__runs.get_list():
            temp_nodes = self.__runs.get_list()
            temp_nodes.remove(run)
            self.__runs = MappedList(temp_nodes)
            success = True
        else:
            raise ValueError(f"run {run.__name} doesn't exists")
        return success

    def is_assigned(self):
        """
        :return: whether the vehicle is assigned to at-least one run
        """
        return self.__runs.size() != 0

    def get_assign(self):
        """
        :return: provides the vehicle assignment
        """
        content = f"Vehicle: {self.__name}\n"
        for run in self.__runs.get_list():
            content += f"{run.__str__()}\n"
        return content
