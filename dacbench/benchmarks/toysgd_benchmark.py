import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import pandas as pd
from gymnasium import spaces

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import ToySGDEnv

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
LR = CSH.UniformFloatHyperparameter(name="0_log_learning_rate", lower=-10, upper=0)
MOMENTUM = CSH.UniformFloatHyperparameter(name="1_log_momentum", lower=-10, upper=0)
DEFAULT_CFG_SPACE.add_hyperparameter(LR)
DEFAULT_CFG_SPACE.add_hyperparameter(MOMENTUM)


INFO = {
    "identifier": "toy_sgd",
    "name": "Learning Rate and Momentum Adaption for SGD on Toy Functions",
    "reward": "Negative Log Regret",
    "state_description": [
        "Remaining Budget",
        "Gradient",
        "Current Learning Rate",
        "Current Momentum",
    ],
    "action_description": ["Log Learning Rate", "Log Momentum"],
}

DEFAULTS = objdict(
    {
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Dict",
        "observation_space_type": None,
        "observation_space_args": [
            {
                "remaining_budget": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "gradient": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                "learning_rate": spaces.Box(low=0, high=1, shape=(1,)),
                "momentum": spaces.Box(low=0, high=1, shape=(1,)),
            }
        ],
        "reward_range": (-np.inf, np.inf),
        "cutoff": 10,
        "seed": 0,
        "multi_agent": False,
        "instance_set_path": "../instance_sets/toysgd/toysgd_default.csv",
        "benchmark_info": INFO,
    }
)


class ToySGDBenchmark(AbstractBenchmark):
    def __init__(self, config_path=None, config=None):
        """
        Initialize SGD Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(ToySGDBenchmark, self).__init__(config_path, config)
        if not self.config:
            self.config = objdict(DEFAULTS.copy())

        for key in DEFAULTS:
            if key not in self.config:
                self.config[key] = DEFAULTS[key]

    def get_environment(self):
        """
        Return SGDEnv env with current configuration

        Returns
        -------
        SGDEnv
            SGD environment
        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        # Read test set if path is specified
        if (
            "test_set" not in self.config.keys()
            and "test_set_path" in self.config.keys()
        ):
            self.read_instance_set(test=True)

        env = ToySGDEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self, test=False):
        """
        Read path of instances from config into list
        """
        if test:
            path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/"
                + self.config.test_set_path
            )
            keyword = "test_set"
        else:
            path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/"
                + self.config.instance_set_path
            )
            keyword = "instance_set"

        self.config[keyword] = {}
        with open(path, "r") as fh:
            # reader = csv.DictReader(fh, delimiter=";")
            df = pd.read_csv(fh, sep=";")
            for index, instance in df.iterrows():
                self.config[keyword][int(instance["ID"])] = instance
