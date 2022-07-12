from base.common.TravelMeasureMatrix import TravelMeasureMatrix
from common.constant.constants import TimeMatrixSource
from common.util.data_util import load_nx_travel_times


class TravelTimeMatrix(TravelMeasureMatrix):
    def __init__(self, travel_time_data_set, selected_agency, selected_date):
        super(TravelMeasureMatrix, self).__init__()
        if travel_time_data_set == TimeMatrixSource.NXN:
            func = load_nx_travel_times
        else:
            raise ValueError(f"Invalid travel-time data set {travel_time_data_set}")
        self._travel_measure = func(
            selected_agency=selected_agency,
            selected_date=selected_date,
        )

    def get_travel_time(self, _loc_i_idx, _loc_j_idx):
        """
        :param _loc_i_idx: index of start location
        :param _loc_j_idx: index of end location
        :return: travel time (IN MINUTES) and missing status
        """
        return self._get_travel_measure(_loc_i_idx, _loc_j_idx)
