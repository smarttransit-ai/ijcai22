from base.data.GenDataLoader import GenDataLoader
from base.errors import ImplementationError
from common.util.common_util import logger


class Request(object):
    """
        class to resemble a real request
    """

    def __init__(
            self, pick_up, drop_off,
            client_id=-1, booking_id=-1, start_time="00:00:00",
            broad_time_windows=None, travel_time=0,
            pick_up_zip=-1, drop_off_zip=-1,
    ):
        self.pick_up_node = pick_up
        self.drop_off_node = drop_off
        self.client_id = client_id
        self.booking_id = booking_id
        self.start_time = start_time
        self.broad_time_windows = broad_time_windows
        self.travel_time = travel_time
        self.pick_up_zip = pick_up_zip
        self.drop_off_zip = drop_off_zip
        self.validate()

    def copy(self):
        request = Request(self.pick_up_node, self.drop_off_node)
        request.client_id = self.client_id
        request.booking_id = self.booking_id
        request.start_time = self.start_time
        request.broad_time_windows = self.broad_time_windows
        request.travel_time = self.travel_time
        request.pick_up_zip = self.pick_up_zip
        request.drop_off_zip = self.drop_off_zip
        return request

    def specify(self, alpha):
        """
        :param alpha: the earliest pickup time
        :return: request with pick-up and drop-off window adjusted
        """
        step_size = GenDataLoader.instance().agency_config.time_window_gap
        travel_time = self.travel_time
        self.pick_up_node.earliest_arrival = alpha
        self.pick_up_node.latest_arrival = alpha + step_size
        self.drop_off_node.earliest_arrival = alpha + travel_time
        self.drop_off_node.latest_arrival = alpha + step_size + travel_time
        return self

    def validate(self):
        """
            Verify whether the nodes are valid
        """
        failure_type_1 = self.pick_up_node.node_type != "pick_up"
        failure_type_2 = self.drop_off_node.node_type != "drop_off"
        failure_type_3 = self.pick_up_node.idx == self.drop_off_node.idx
        failure_type_4 = self.pick_up_node.earliest_arrival > self.drop_off_node.latest_arrival
        failure_type_5 = self.pick_up_node.capacity != abs(self.drop_off_node.capacity)
        failure_type_6 = self.pick_up_node.capacity == 0
        failure_type_7 = self.drop_off_node.capacity == 0

        if failure_type_1 or failure_type_2 or failure_type_3 or failure_type_4 or \
                failure_type_5 or failure_type_6 or failure_type_7:
            if failure_type_1:
                logger.error(f"First node should be pick-up node,"
                             f" where the the current one is of type {self.pick_up_node.node_type}")
            if failure_type_2:
                logger.error(f"Second node should be drop-off node,"
                             f" where the the current one is of type {self.drop_off_node.node_type}")
            if failure_type_3:
                logger.error(f"Both nodes have same index, the index is {self.pick_up_node.idx}")
            if failure_type_4:
                logger.error(f"Earliest arrival of pick-up node ({self.pick_up_node.earliest_arrival}) should be come "
                             f"before latest arrival of drop-off node ({self.drop_off_node.latest_arrival})")
            if failure_type_5:
                logger.error(f"Pick-up capacity {self.pick_up_node.capacity} and "
                             f"drop-off capacity {abs(self.drop_off_node.capacity)} are different")
            if failure_type_6:
                logger.error(f"Pick-up capacity is zero")
            if failure_type_7:
                logger.error(f"Drop-off capacity is zero")

            raise ImplementationError("Error in implementations")

    def __str__(self):
        return f"{self.client_id}-{self.booking_id}-{self.pick_up_node.__str__(), self.drop_off_node.__str__()}"
