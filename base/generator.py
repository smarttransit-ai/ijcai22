from base.data.GenDataLoader import GenDataLoader
from common.constant.configs import DataConfiguration, SampleInstanceConfiguration
from common.constant.constants import TimeMatrixSource, get_time_matrix_src


def get_configs(
        exact_passenger_count=True,
        skip_dump_process=False,
        travel_time_data_set=TimeMatrixSource.NXN,
        selected_agency="AGENCY_A",
        selected_date=0,
        random_seed=0,
        no_of_trips=-1,
        no_of_runs=-1,
        no_of_vehicles=-1,
        vehicle_capacity=-1,
        dwell_time=-1,
        drop_off_flexibility=-1,
        broad_time_window_gap=-1,
        custom_args=None
):
    """
    this function used to generate input data configuration and sample instance configuration
    :return: the data configuration and configuration of the sample instance
    """
    if custom_args is not None:
        no_of_trips = int(custom_args.no_of_trips)
        no_of_runs = int(custom_args.no_of_runs)
        no_of_vehicles = int(custom_args.no_of_vehicles)
        selected_date = custom_args.date
        selected_agency = custom_args.agency
        vehicle_capacity = int(custom_args.vehicle_capacity)
        dwell_time = int(custom_args.dwell_time)
        drop_off_flexibility = int(custom_args.drop_off_flexibility)
        broad_time_window_gap = int(custom_args.broad_time_window_gap)
        travel_time_data_set = get_time_matrix_src(custom_args.time_matrix_src)
        if custom_args.exact_passenger_count == "True":
            exact_passenger_count = True
        else:
            exact_passenger_count = False

    data_config = DataConfiguration(
        exact_passenger_count=exact_passenger_count,
        skip_dump_process=skip_dump_process,
        travel_time_data_set=travel_time_data_set
    )

    sample_ins_config = SampleInstanceConfiguration(
        selected_agency=selected_agency,
        selected_date=selected_date,
        random_seed=random_seed,
        no_of_trips=no_of_trips,
        no_of_runs=no_of_runs,
        no_of_vehicles=no_of_vehicles,
        vehicle_capacity=vehicle_capacity,
        dwell_time=dwell_time,
        drop_off_flexibility=drop_off_flexibility,
        broad_time_window_gap=broad_time_window_gap
    )

    GenDataLoader.instance().load(args=(sample_ins_config, data_config, None))

    return data_config, sample_ins_config


def create_real(
        exact_passenger_count=True,
        skip_dump_process=False,
        travel_time_data_set=TimeMatrixSource.NXN,
        selected_agency="AGENCY_A",
        selected_date=0,
        random_seed=0,
        no_of_trips=-1,
        no_of_runs=-1,
        no_of_vehicles=-1,
        vehicle_capacity=-1,
        dwell_time=-1,
        drop_off_flexibility=-1,
        broad_time_window_gap=-1,
        booking_ids=None,
        custom_args=None
):
    """
    this function creates the sample real assignment instance given the following configurations

    :return: the sample greedy instance
    """
    from algo.heuristic.RealAssignPTOpt import RealAssignPTOpt
    data_config, sample_ins_config = get_configs(
        exact_passenger_count=exact_passenger_count,
        skip_dump_process=skip_dump_process,
        travel_time_data_set=travel_time_data_set,
        selected_agency=selected_agency,
        selected_date=selected_date,
        random_seed=random_seed,
        no_of_trips=no_of_trips,
        no_of_runs=no_of_runs,
        no_of_vehicles=no_of_vehicles,
        vehicle_capacity=vehicle_capacity,
        dwell_time=dwell_time,
        drop_off_flexibility=drop_off_flexibility,
        broad_time_window_gap=broad_time_window_gap,
        custom_args=custom_args
    )
    return RealAssignPTOpt(
        sample_inst_config=sample_ins_config, config=data_config, custom_args=custom_args, booking_ids=booking_ids
    )


def create_routing(
        exact_passenger_count=True,
        skip_dump_process=False,
        travel_time_data_set=TimeMatrixSource.NXN,
        selected_agency="AGENCY_A",
        selected_date=0,
        random_seed=0,
        no_of_trips=-1,
        no_of_runs=-1,
        no_of_vehicles=-1,
        vehicle_capacity=-1,
        dwell_time=-1,
        drop_off_flexibility=-1,
        broad_time_window_gap=-1,
        custom_args=None
):
    """
    this function creates the routing wrapper implementation
    :return: the routing wrapper instance
    """
    from algo.baseline.RoutingBasedPTOpt import RoutingBasedPTOpt
    from base.entity.Run import Run
    data_config, sample_ins_config = get_configs(
        exact_passenger_count=exact_passenger_count,
        skip_dump_process=skip_dump_process,
        travel_time_data_set=travel_time_data_set,
        selected_agency=selected_agency,
        selected_date=selected_date,
        random_seed=random_seed,
        no_of_trips=no_of_trips,
        no_of_runs=no_of_runs,
        no_of_vehicles=no_of_vehicles,
        vehicle_capacity=vehicle_capacity,
        dwell_time=dwell_time,
        drop_off_flexibility=drop_off_flexibility,
        broad_time_window_gap=broad_time_window_gap,
        custom_args=custom_args
    )
    return RoutingBasedPTOpt(
        run_cls=Run, sample_inst_config=sample_ins_config, config=data_config, custom_args=custom_args
    )


def create_vroom(
        exact_passenger_count=True,
        skip_dump_process=False,
        travel_time_data_set=TimeMatrixSource.NXN,
        selected_agency="AGENCY_A",
        selected_date=0,
        random_seed=0,
        no_of_trips=-1,
        no_of_runs=-1,
        no_of_vehicles=-1,
        vehicle_capacity=-1,
        dwell_time=-1,
        drop_off_flexibility=-1,
        broad_time_window_gap=-1,
        custom_args=None
):
    """
    this function creates the vroom wrapper instance
    :return: the vroom wrapper instance
    """
    from algo.baseline.VRoomBasedPTOpt import VRoomBasedPTOpt
    from base.entity.Run import Run
    data_config, sample_ins_config = get_configs(
        exact_passenger_count=exact_passenger_count,
        skip_dump_process=skip_dump_process,
        travel_time_data_set=travel_time_data_set,
        selected_agency=selected_agency,
        selected_date=selected_date,
        random_seed=random_seed,
        no_of_trips=no_of_trips,
        no_of_runs=no_of_runs,
        no_of_vehicles=no_of_vehicles,
        vehicle_capacity=vehicle_capacity,
        dwell_time=dwell_time,
        drop_off_flexibility=drop_off_flexibility,
        broad_time_window_gap=broad_time_window_gap,
        custom_args=custom_args
    )
    return VRoomBasedPTOpt(
        run_cls=Run, sample_inst_config=sample_ins_config, config=data_config, custom_args=custom_args
    )


def create_greedy(
        exact_passenger_count=True,
        skip_dump_process=False,
        travel_time_data_set=TimeMatrixSource.NXN,
        selected_agency="AGENCY_A",
        selected_date=0,
        random_seed=0,
        no_of_trips=-1,
        no_of_runs=-1,
        no_of_vehicles=-1,
        vehicle_capacity=-1,
        dwell_time=-1,
        drop_off_flexibility=-1,
        broad_time_window_gap=-1,
        custom_args=None
):
    """
    this function creates the basic greedy instance
    :return: the sample greedy instance
    """
    from algo.heuristic.GreedyPTOpt import GreedyPTOpt
    from base.entity.Run import Run
    data_config, sample_ins_config = get_configs(
        exact_passenger_count=exact_passenger_count,
        skip_dump_process=skip_dump_process,
        travel_time_data_set=travel_time_data_set,
        selected_agency=selected_agency,
        selected_date=selected_date,
        random_seed=random_seed,
        no_of_trips=no_of_trips,
        no_of_runs=no_of_runs,
        no_of_vehicles=no_of_vehicles,
        vehicle_capacity=vehicle_capacity,
        dwell_time=dwell_time,
        drop_off_flexibility=drop_off_flexibility,
        broad_time_window_gap=broad_time_window_gap,
        custom_args=custom_args
    )
    return GreedyPTOpt(
        run_cls=Run, sample_inst_config=sample_ins_config, config=data_config, custom_args=custom_args
    )


def create_sim_anneal(
        exact_passenger_count=True,
        skip_dump_process=False,
        travel_time_data_set=TimeMatrixSource.NXN,
        selected_agency="AGENCY_A",
        selected_date=0,
        random_seed=0,
        no_of_trips=-1,
        no_of_runs=-1,
        no_of_vehicles=-1,
        vehicle_capacity=-1,
        dwell_time=-1,
        drop_off_flexibility=-1,
        broad_time_window_gap=-1,
        custom_args=None
):
    """
    this function creates a simulated annealing instance
    :return: the simulated annealing instance
    """
    from algo.heuristic.GreedyPTOpt import GreedyPTOpt
    from algo.heuristic.SABasedPTOpt import SABasedPTOpt
    from base.entity.Run import Run
    data_config, sample_ins_config = get_configs(
        exact_passenger_count=exact_passenger_count,
        skip_dump_process=skip_dump_process,
        travel_time_data_set=travel_time_data_set,
        selected_agency=selected_agency,
        selected_date=selected_date,
        random_seed=random_seed,
        no_of_trips=no_of_trips,
        no_of_runs=no_of_runs,
        no_of_vehicles=no_of_vehicles,
        vehicle_capacity=vehicle_capacity,
        dwell_time=dwell_time,
        drop_off_flexibility=drop_off_flexibility,
        broad_time_window_gap=broad_time_window_gap,
        custom_args=custom_args
    )
    return SABasedPTOpt(
        run_cls=Run, greedy_cls=GreedyPTOpt,
        sample_inst_config=sample_ins_config, config=data_config, custom_args=custom_args
    )


def create_online_booking_simulation(
        exact_passenger_count=True,
        skip_dump_process=False,
        travel_time_data_set=TimeMatrixSource.NXN,
        selected_agency="AGENCY_A",
        selected_date=0,
        random_seed=0,
        no_of_trips=-1,
        no_of_runs=-1,
        no_of_vehicles=-1,
        vehicle_capacity=-1,
        dwell_time=-1,
        drop_off_flexibility=-1,
        broad_time_window_gap=-1,
        custom_args=None
):
    """
    this function creates a mixed_solver
    :return: the agent_with_anytime solver
    """
    from algo.heuristic.GreedyPTOpt import GreedyPTOpt
    from algo.heuristic.OnlineBookingSimulator import OnlineBookingSimulator
    from base.entity.Run import Run
    # this will be always run in real-time mode
    data_config, sample_ins_config = get_configs(
        exact_passenger_count=exact_passenger_count,
        skip_dump_process=skip_dump_process,
        travel_time_data_set=travel_time_data_set,
        selected_agency=selected_agency,
        selected_date=selected_date,
        random_seed=random_seed,
        no_of_trips=no_of_trips,
        no_of_runs=no_of_runs,
        no_of_vehicles=no_of_vehicles,
        vehicle_capacity=vehicle_capacity,
        dwell_time=dwell_time,
        drop_off_flexibility=drop_off_flexibility,
        broad_time_window_gap=broad_time_window_gap,
        custom_args=custom_args
    )
    return OnlineBookingSimulator(
        rt_solver_cls=GreedyPTOpt, run_cls=Run,
        sample_inst_config=sample_ins_config,
        config=data_config, custom_args=custom_args
    )


def create_vroom_online_booking_simulation(
        exact_passenger_count=True,
        skip_dump_process=False,
        travel_time_data_set=TimeMatrixSource.NXN,
        selected_agency="AGENCY_A",
        selected_date=0,
        random_seed=0,
        no_of_trips=-1,
        no_of_runs=-1,
        no_of_vehicles=-1,
        vehicle_capacity=-1,
        dwell_time=-1,
        drop_off_flexibility=-1,
        broad_time_window_gap=-1,
        custom_args=None
):
    """
    this function creates a mixed_solver
    :return: the agent_with_anytime solver
    """
    from algo.baseline.VRoomBasedPTOpt import VRoomBasedPTOpt
    from algo.heuristic.OnlineBookingSimulator import BaselineBasedOnlineBookingSimulator
    from base.entity.Run import Run
    # this will be always run in real-time mode
    data_config, sample_ins_config = get_configs(
        exact_passenger_count=exact_passenger_count,
        skip_dump_process=skip_dump_process,
        travel_time_data_set=travel_time_data_set,
        selected_agency=selected_agency,
        selected_date=selected_date,
        random_seed=random_seed,
        no_of_trips=no_of_trips,
        no_of_runs=no_of_runs,
        no_of_vehicles=no_of_vehicles,
        vehicle_capacity=vehicle_capacity,
        dwell_time=dwell_time,
        drop_off_flexibility=drop_off_flexibility,
        broad_time_window_gap=broad_time_window_gap,
        custom_args=custom_args
    )
    return BaselineBasedOnlineBookingSimulator(
        rt_solver_cls=VRoomBasedPTOpt, run_cls=Run,
        sample_inst_config=sample_ins_config,
        config=data_config, custom_args=custom_args
    )


def create_routing_online_booking_simulator(
        exact_passenger_count=True,
        skip_dump_process=False,
        travel_time_data_set=TimeMatrixSource.NXN,
        selected_agency="AGENCY_A",
        selected_date=0,
        random_seed=0,
        no_of_trips=-1,
        no_of_runs=-1,
        no_of_vehicles=-1,
        vehicle_capacity=-1,
        dwell_time=-1,
        drop_off_flexibility=-1,
        broad_time_window_gap=-1,
        custom_args=None
):
    """
    this function creates a mixed_solver
    :return: the agent_with_anytime solver
    """
    from algo.baseline.RoutingBasedPTOpt import RoutingBasedPTOpt
    from algo.heuristic.OnlineBookingSimulator import BaselineBasedOnlineBookingSimulator
    from base.entity.Run import Run
    # this will be always run in real-time mode
    data_config, sample_ins_config = get_configs(
        exact_passenger_count=exact_passenger_count,
        skip_dump_process=skip_dump_process,
        travel_time_data_set=travel_time_data_set,
        selected_agency=selected_agency,
        selected_date=selected_date,
        random_seed=random_seed,
        no_of_trips=no_of_trips,
        no_of_runs=no_of_runs,
        no_of_vehicles=no_of_vehicles,
        vehicle_capacity=vehicle_capacity,
        dwell_time=dwell_time,
        drop_off_flexibility=drop_off_flexibility,
        broad_time_window_gap=broad_time_window_gap,
        custom_args=custom_args
    )
    return BaselineBasedOnlineBookingSimulator(
        rt_solver_cls=RoutingBasedPTOpt, run_cls=Run,
        sample_inst_config=sample_ins_config,
        config=data_config, custom_args=custom_args
    )
