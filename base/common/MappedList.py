class MappedList(object):
    """
        Implementation of Mapped List for ease of access
    """

    def __init__(self, inp_list=None):
        self.__list = []
        self.__map = {}
        if inp_list is not None:
            self.__list = inp_list
            for i, value in enumerate(self.__list):
                if self.contains(value):
                    raise ValueError(f"{value} is already presents in the list {self.__list}")
                self.__map[value] = i

    def copy(self):
        return MappedList(self.__list)

    def contains(self, value):
        """
        This is pseudo implementation to use .contains() call
        :param value: value
        :return: whether the value in the list or not
        """
        contains = True
        try:
            self.__map[value]
        except KeyError:
            contains = False
        return contains

    def index(self, value):
        """
        This is pseudo implementation to use .index() call
        :param value: value
        :return: the index of the value
        """
        if self.contains(value):
            return self.__map[value]
        raise ValueError(f"value {value} not exists in the list {self.__list}")

    def get_list(self):
        """
        :return: the original list
        """
        return self.__list

    def size(self):
        """
        :return: the size of the list
        """
        return len(self.__list)
