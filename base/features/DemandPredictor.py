import pandas as pd

from common.util.common_util import read_data_file


class DemandPredictor(object):
    def __init__(self, agency):

        def basic_data_process():
            _df = read_data_file(agency, date_filter_off=True)
            _df["date"] = pd.to_datetime(_df["Date"], format="%Y-%m-%d")
            _df["day_of_week"] = _df["date"].dt.dayofweek
            _df["time"] = pd.to_datetime(_df["Sch Time in HH:MM:SS"], format="%H:%M:%S")
            _df["hour"] = _df.time.dt.hour
            return _df

        df = basic_data_process()

        def load_zone_mix_data(_df):
            _df = _df.groupby(["Pickup Zip", "Dropoff Zip", "hour", "day_of_week"]).size().reset_index(name="count")
            _df["count"] /= len(_df)
            return _df

        pdh_df = load_zone_mix_data(df)
        self.pdh_demand_dict = {}
        for i, entry in pdh_df.iterrows():
            self.pdh_demand_dict[
                int(entry["Pickup Zip"]), int(entry["Dropoff Zip"]), int(entry["hour"]), int(entry["day_of_week"])] \
                = float(entry["count"])

        def load_p_zone_mix_data(_df):
            _df = _df.groupby(["Pickup Zip", "hour", "day_of_week"]).size().reset_index(name="count")
            _df["count"] /= len(_df)
            return _df

        ph_df = load_p_zone_mix_data(df)
        self.ph_demand_dict = {}
        for i, entry in ph_df.iterrows():
            self.ph_demand_dict[int(entry["Pickup Zip"]), int(entry["hour"]), int(entry["day_of_week"])] = \
                float(entry["count"])

        def load_d_zone_mix_data(_df):
            _df = _df.groupby(["Dropoff Zip", "hour", "day_of_week"]).size().reset_index(name="count")
            _df["count"] /= len(_df)
            return _df

        dh_df = load_d_zone_mix_data(df)
        self.dh_demand_dict = {}
        for i, entry in dh_df.iterrows():
            self.dh_demand_dict[int(entry["Dropoff Zip"]), int(entry["hour"]), int(entry["day_of_week"])] = \
                float(entry["count"])

        def load_hour_dow_data(_df):
            _df = _df.groupby(["hour", "day_of_week"]).size().reset_index(name="count")
            _df["count"] /= len(_df)
            return _df

        h_df = load_hour_dow_data(df)
        self.h_demand_dict = {}
        for i, entry in h_df.iterrows():
            self.h_demand_dict[int(entry["hour"]), int(entry["day_of_week"])] = float(entry["count"])
