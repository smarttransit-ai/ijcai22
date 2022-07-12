from base.data.GenericData import GenericData


class CSData(GenericData):
    """
        This class stores custom data from para-transit dataset for Custom Solver
    """

    def __init__(
            self,
            sample_inst_config=None,
            config=None,
            custom_agency_config=None,
    ):
        super(CSData, self).__init__(sample_inst_config, config, custom_agency_config)

    def get_data_model(self):
        data = {
            'pick_up_nodes': self.pick_up_nodes,
            'drop_off_nodes': self.drop_off_nodes,
            'pick_up_time_windows': self.pick_up_time_windows,
            'pick_up_broad_time_windows': self.pick_up_broad_time_windows,
            'travel_times': self.travel_times,
            'drop_off_time_windows': self.drop_off_time_windows,
            'true_locations': self.true_locations,
            'capacities': self.capacities,
            'wheel_chair_counts': self.wheel_chair_counts,
            'client_ids': self.client_ids,
            'booking_ids': self.booking_ids,
            'start_times': self.scheduled_times,
            'pick_up_zips': self.pick_up_zips,
            'drop_off_zips': self.drop_off_zips,
        }
        return data
