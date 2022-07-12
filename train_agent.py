########################################
#              TRAIN AGENT             #
########################################

import logging
import warnings
from ast import literal_eval

from base.learn.RealTimeEnv import RealTimeEnv, RealTimeAnyTimeEnv
from base.learn.RealTimeRLAgent import RealTimeRLAgentNN
from common.constant.constants import RESULT_DIR
from common.parsers.ArgParser import CustomArgParser
from common.util.common_util import logger

warnings.filterwarnings('ignore')
logger.setLevel(logging.CRITICAL)

arg_parser = CustomArgParser()
args = arg_parser.parse_args()

if not isinstance(args.features, list):
    features = literal_eval(args.features)
else:
    features = args.features

features_size = len(features)
if features_size == 0:
    features_size = 8

train_agent_model = str(args.train_agent_model)
train_env_type = str(args.train_env_type)

agent_class = RealTimeRLAgentNN


simple_agent = agent_class(
    state_space=features_size,
    memory_file_name=f"{RESULT_DIR}/{args.agency}/models/memory_{args.random_seed}.csv"
)

env_dict = {
    "agent_with_anytime": RealTimeAnyTimeEnv,
    "agent_without_anytime": RealTimeEnv
}
simple_agent.train(env_dict, args)
