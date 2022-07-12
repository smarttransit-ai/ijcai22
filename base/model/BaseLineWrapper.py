##################################################################
#              BASE LINE COMMON WRAPPER IMPLEMENTATION           #
##################################################################
import sys

from base.data.GenDataLoader import GenDataLoader
from base.model.CSBasePTOpt import CSBasePTOpt
from common.constant.constants import MIN_SEC_SWAP, CSV_EXTENSION, DATA_DIR
from common.util.common_util import logger, file_exists


class BaseLineWrapper(CSBasePTOpt):
    """
        This is a wrapper class for solving baseline VRP solutions
    """

    def __init__(self, prefix, run_cls, sample_inst_config=None, config=None, custom_args=None):
        super(BaseLineWrapper, self).__init__(prefix, sample_inst_config, config, custom_args)
        self._run_cls = run_cls
        self.use_custom_tight_window = False
        """
            by default, the agent expect an environment that runs anytime algorithm in between,
            thus the trainer is set as ``agent_with_anytime'' it is also possible to run without anytime algorithm, 
            in that case it will use the model that trained without running the anytime algorithm in between
        """
        self.trainer = "agent_with_anytime"

    def assign_requests(self, idx=-1):
        raise NotImplementedError

    def assign_requests_parallel(self, idx=-1):
        """
            Since the Baseline VRP solvers has limited control, implementing ``assign request'' using parallelism is not
            possible from our end, but there are in built parallelism possible for baseline solver that can be used
            instead
        """
        return self.assign_requests(idx=idx)

    def solve(self, write_summary=False):
        # before solving make sure the correct filtering applied
        if self.use_custom_tight_window:
            import pandas as pd
            broad_window = int(GenDataLoader.instance().agency_config.broad_time_window_gap / MIN_SEC_SWAP)
            file_name = f"{DATA_DIR}/{self._args.agency}/tight_windows/{self.trainer}/{broad_window}/{self._args.date}"
            if file_exists(file_name, CSV_EXTENSION):
                df = pd.read_csv(file_name + CSV_EXTENSION)
                request_dict = {}
                for i, req in enumerate(self.requests):
                    request_dict[req.booking_id] = i
                for k, entry in df.iterrows():
                    booking_id = int(entry["booking_id"])
                    if booking_id in request_dict.keys():
                        idx = request_dict[booking_id]
                        request = self.requests[idx]
                        request = request.specify(int(entry['p_earl_t']))
                        self.requests[idx] = request
                super(BaseLineWrapper, self).solve(write_summary)
            else:
                logger.error(f"The tight-window configuration file: {file_name + CSV_EXTENSION} not present !!!")
                sys.exit(-1)
        else:
            super(BaseLineWrapper, self).solve(write_summary)
