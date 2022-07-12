class DoubleMap(object):
    """
      Implementation of two side map to avoid multiple lookups
      this implementation is only support when the mapping is one to one

      Set of Keys and Set of Values need to be disjoint
    """

    def __init__(self):
        self._forward = {}
        self._reverse = {}

    def copy(self):
        dbl_map = DoubleMap()
        dbl_map._forward = self._forward.copy()
        dbl_map._reverse = self._reverse.copy()
        return dbl_map

    def insert(self, key, value):
        """
        this function will insert the key-value pair to the forward map and value-key pair to backward map
        :param key: key
        :param value: value
        """
        if key == value:
            raise ValueError(f"{key} and {value} are same")
        if key in self._forward.keys():
            raise KeyError(f"{key} is already exists")
        elif value in self._forward.keys():
            raise KeyError(f"{value} is already exists")
        if value in self._reverse.keys():
            raise KeyError(f"{value} is already exists")
        elif key in self._reverse.keys():
            raise KeyError(f"{key} is already exists")
        self._forward[key] = value
        self._reverse[value] = key

    def remove(self, key):
        """
        this function will remove the key-value pair in the forward map and value-key pair in the backward map
        :param key: key
        """
        if key in self._forward.keys():
            value = self._forward[key]
            self._forward.pop(key)
            self._reverse.pop(value)
        elif key in self._reverse.keys():
            value = self._reverse[key]
            self._reverse.pop(key)
            self._forward.pop(value)
        else:
            raise KeyError(f"{key} is not available")

    def get(self, key):
        """
        this function will returns the key-value pair, when either key or value is requested
        :param key: key
        """
        if key in self._forward.keys():
            response = key, self._forward[key]
        elif key in self._reverse.keys():
            response = self._reverse[key], key
        else:
            raise KeyError(f"{key} is not available")
        return response


class RequestDoubleMap(DoubleMap):
    """
         Custom implementation of two side map to avoid multiple lookups for request
         this implementation is only support when the mapping is one to one

         Set of Keys and Set of Values need to be disjoint
       """

    def __init__(self):
        super(RequestDoubleMap, self).__init__()
        self.__request_map = {}

    def copy(self):
        dbl_map = RequestDoubleMap()
        dbl_map._forward = self._forward.copy()
        dbl_map._reverse = self._reverse.copy()
        dbl_map.__request_map = self.__request_map.copy()
        return dbl_map

    def insert_request(self, request):
        if request.pick_up_node.idx in self.__request_map.keys():
            raise KeyError(f"{request.pick_up_node.idx} is already exists")
        if request.drop_off_node.idx in self.__request_map.keys():
            raise KeyError(f"{request.drop_off_node.idx} is already exists")
        self.__request_map[request.pick_up_node.idx] = request
        self.__request_map[request.drop_off_node.idx] = request
        self.insert(request.pick_up_node.idx, request.drop_off_node.idx)

    def remove_request(self, request):
        """
        this will remove the entire request
        """
        self.remove(request.pick_up_node.idx)

    def remove(self, key):
        """
        this function will remove the key-value pair in the forward map and value-key pair in the backward map
        :param key: key
        """
        if key in self._forward.keys():
            value = self._forward[key]
            self._forward.pop(key)
            self._reverse.pop(value)
            self.__request_map.pop(key)
            self.__request_map.pop(value)
        elif key in self._reverse.keys():
            value = self._reverse[key]
            self._reverse.pop(key)
            self._forward.pop(value)
            self.__request_map.pop(key)
            self.__request_map.pop(value)
        else:
            raise KeyError(f"{key} is not available")

    def get_request(self, node):
        """
        this function will returns the key-value pair, when either key or value is requested
        :param node: pick-up or drop-off node
        """
        key = node.idx
        if key in self.__request_map.keys():
            request = self.__request_map[key]
        else:
            request = None
        return request
