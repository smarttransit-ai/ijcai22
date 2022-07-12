import math


class TravelMeasureMatrix(object):
    def __init__(self):
        self._travel_measure = {}

    def size(self):
        """
        :return: the size of the matrix
        """
        return len(self._travel_measure)

    def keys(self):
        """
        :return: the keys of the matrix
        """
        return self._travel_measure.keys()

    def _get_travel_measure(self, _loc_i_idx, _loc_j_idx):
        """
        :param _loc_i_idx: index of start location
        :param _loc_j_idx: index of end location
        :return: travel measure (IN GIVEN UNIT) and missing status
        """
        if (_loc_i_idx, _loc_j_idx) in self._travel_measure.keys():
            travel_measure = self._travel_measure[(_loc_i_idx, _loc_j_idx)]
            missing = False
        else:
            travel_measure = math.inf
            missing = True
        return missing, travel_measure

    def compare_another_matrix(self, matrix):
        """
        :param matrix: another travel-time matrix with different source
        print the comparison statistics
        """
        from pandas import DataFrame
        df = DataFrame()
        ratios = []
        for (i, j) in self._travel_measure.keys():
            if (i, j) in matrix.keys():
                f_tt = self._travel_measure[(i, j)]
                _, s_tt = matrix.get_travel_measure(i, j)
                if s_tt != 0:
                    ratios.append(f_tt / s_tt)
        df["ratio"] = ratios
        print(df.describe())

    def get_matrix(self):
        """
        :return: entire travel measure matrix
        """
        return self._travel_measure.copy()
