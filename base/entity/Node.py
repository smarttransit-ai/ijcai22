from enum import Enum

from base.data.GenDataLoader import GenDataLoader
from common.util.common_util import convert_sec_to_hh_mm_ss


class NodeType(Enum):
    SOURCE = 'source'
    SINK = 'sink'
    PICK_UP = 'pick_up'
    DROP_OFF = 'drop_off'


class Node(object):
    """
        This class stores the details of Node
    """

    def __init__(
            self, idx, real_location, earliest_arrival, latest_arrival, node_type, wheel_chair_count=None, capacity=0
    ):
        # index of node, used for ver
        self.idx = idx

        # true spatial location in the format (latitude, longitude)
        self.real_location = real_location

        # earliest time to reach the node
        self.earliest_arrival = earliest_arrival

        # latest time to reach the node
        self.latest_arrival = latest_arrival

        # type of the node
        self.node_type = node_type

        # number of wheel-chair
        self.wheel_chair_count = wheel_chair_count

        # capacity required
        self.capacity = capacity

    def reachable(self, node_y):
        """
        :param node_y: node to which the current node reach
        :return: whether this node can reach the node-y
        """
        _, travel_time = GenDataLoader.instance().get_travel_time(self.real_location, node_y.real_location)
        if self.node_type == NodeType.SOURCE.value:
            if node_y.node_type != NodeType.PICK_UP.value:
                return False
        elif self.node_type == NodeType.PICK_UP.value:
            if node_y.node_type == NodeType.SOURCE.value or node_y.node_type == NodeType.SINK.value:
                return False
        elif self.node_type == NodeType.DROP_OFF.value:
            if node_y.node_type == NodeType.SOURCE.value:
                return False
        else:
            return False
        return self.reachable_inner(node_y, travel_time)

    def reachable_inner(self, node_y, travel_time):
        """
        :param node_y: node to which the current node reach
        :param travel_time: travel time
        :return: whether this node can reach the node-y
        """
        # dwell time for the passenger pick-up and drop-off
        dwell_time = GenDataLoader.instance().agency_config.dwell_time

        # additional detour time, that the passenger can tolerate
        drop_off_flexibility = GenDataLoader.instance().agency_config.drop_off_flexibility

        # maximum duration, to reach
        upper_limit = node_y.latest_arrival
        if node_y.node_type == NodeType.DROP_OFF.value:
            upper_limit += drop_off_flexibility

        return (self.earliest_arrival + travel_time + dwell_time <= upper_limit) and (self != node_y)

    def __str__(self):
        return f"{self.idx} {convert_sec_to_hh_mm_ss(self.earliest_arrival), convert_sec_to_hh_mm_ss(self.latest_arrival)}"
