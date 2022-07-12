from argparse import ArgumentParser


class CustomArgParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self)
        self.add_argument("--algo", help="Algo", default="greedy", required=False)
        self.add_argument("--date", help="Selected Date", default=0, type=int, required=False)
        self.add_argument("--data_partition_idx", help="Data Partition Idx", type=int, default=-1, required=False)
        self.add_argument("--agency", help="Selected Agency", default="AGENCY_A", required=False)
        self.add_argument("--data_file", help="Data File", default=None, required=False)
        self.add_argument("--random_seed", help="Random Seed", default=0, type=int, required=False)
        self.add_argument("--model_index", help="Model Index", default=-1, type=int, required=False)
        self.add_argument("--features", help="Features", default=[], required=False)
        self.add_argument("--zero_positions", help="Zero Position", default=[], required=False)
        self.add_argument("--time_matrix_src", help="Time Matrix Source", default="nxn", required=False)
        self.add_argument("--no_of_trips", help="Number of Trips", default=-1, type=int, required=False)
        self.add_argument("--no_of_runs", help="Number of Runs", default=-1, type=int, required=False)
        self.add_argument("--no_of_vehicles", help="Number of Vehicles", default=-1, type=int, required=False)
        self.add_argument("--wait_time", help="Wait time parameter", default=0.01, type=float, required=False)
        self.add_argument(
            "--assign_fraction_cost", help="Assign fraction cost parameter", default=0.01, type=float, required=False
        )
        self.add_argument("--score_threshold", help="Score threshold parameter", default=0.01, type=float, required=False)
        self.add_argument(
            "--assign_fraction_threshold", help="Assign fraction threshold parameter", default=0.01, type=float, required=False
        )
        self.add_argument("--length_of_run", help="Length of run", default=10, type=int, required=False)
        self.add_argument("--soft_max_param", help="Soft max beta parameter", default=0.01, type=float, required=False)
        self.add_argument("--no_of_workers", help="Number of Workers", default=1, type=int, required=False)
        self.add_argument("--population_count", help="Population Count", default=30, type=int, required=False)
        self.add_argument("--filter_pop_count", help="Filter Population Count", default=25, type=int, required=False)
        self.add_argument("--filter_by_real", help="Filter By Real", default="False", required=False)
        self.add_argument("--consider_neg_wait", help="Consider Negative wait time", default="False", required=False)
        self.add_argument("--vehicle_capacity", help="Vehicle Capacity", default=-1, type=int, required=False)
        self.add_argument("--dwell_time", help="Dwell Time", default=-1, type=int, required=False)
        self.add_argument("--drop_off_flexibility", help="Drop-off Flexibility", default=-1, type=int, required=False)
        self.add_argument("--broad_time_window_gap", help="Broad Window Gap", default=-1, type=int, required=False)
        self.add_argument("--exact_passenger_count", help="Consider Exact Passenger", default="True", required=False)
        self.add_argument("--plan_duration", help="Plan Duration", default=3, type=int, required=False)
        self.add_argument("--anytime_plan_duration", help="Anytime Plan Duration", default=3, type=int, required=False)
        self.add_argument("--start_prob", help="Starting Probability", default=0.9, type=float,  required=False)
        self.add_argument("--end_prob", help="Ending Probability", default=0.8, type=float, required=False)
        self.add_argument("--alter_prob", help="Alteration Probability", default=0.2, type=float, required=False)
        self.add_argument("--anytime_algo", help="Anytime algo", default="default", required=False)
        self.add_argument("--train_agent_model", help="Training Agent Model", default="nn", required=False)
        self.add_argument(
            "--train_env_type", help="Training Environment Type", default="agent_with_anytime", required=False
        )
        self.add_argument("--no_of_episodes", help="Number of Episodes", default=2, type=int, required=False)
