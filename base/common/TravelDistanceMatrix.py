from base.common.TravelMeasureMatrix import TravelMeasureMatrix
from common.util.data_util import load_nx_travel_distance


class TravelDistanceMatrix(TravelMeasureMatrix):
    def __init__(self, selected_agency, selected_date):
        super(TravelMeasureMatrix, self).__init__()
        self._travel_measure = load_nx_travel_distance(
            selected_agency=selected_agency,
            selected_date=selected_date,
        )

    def get_distance(self, _loc_i_idx, _loc_j_idx):
        """
        :param _loc_i_idx: index of start location
        :param _loc_j_idx: index of end location
        :return: distance (IN METERS) and missing status
        """
        return self._get_travel_measure(_loc_i_idx, _loc_j_idx)
