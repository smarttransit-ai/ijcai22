import pandas as pd

from common.util.common_util import read_data_file


class ExpectedReqGenerator(object):
    def __init__(self, agency):
        self.expected_req = {}

        def basic_data_process():
            _df = read_data_file(agency, date_filter_off=True)
            _df["date"] = pd.to_datetime(_df["Date"], format="%Y-%m-%d")
            _df["day_of_week"] = _df["date"].dt.dayofweek
            return _df

        df = basic_data_process()

        def get_stats(_df, _i):
            df_weekday = df[df.day_of_week == i]
            weekday_req = len(df_weekday)
            weekday_count = len(df_weekday['date'].unique())
            average_req_per_day = round((weekday_req / weekday_count), 6)
            self.expected_req[i] = average_req_per_day

        for i in range(7):
            get_stats(df, i)
