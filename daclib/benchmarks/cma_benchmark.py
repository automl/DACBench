from daclib.abstract_benchmark import AbstractBenchmark, objdict
from daclib.envs import CMAESEnv

from gym import spaces
import numpy as np
import os
import csv

HISTORY_LENGTH = 5

# TODO: fix this
CMAES_DEFAULTS = objdict(
    {
        "action_space": "Discrete",
        "action_space_args": [int(np.log2(MAX_STEPS))],
        "observation_space": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [
            np.array([-1 for _ in range(HISTORY_LENGTH + 1)]),
            np.array([2 ** max(LUBY_SEQUENCE + 1) for _ in range(HISTORY_LENGTH + 1)]),
        ],
        "reward_range": (-1, 0),
        "cutoff": MAX_STEPS,
        "hist_length": HISTORY_LENGTH,
        "seed": 0,
        #"instance_set_path": "../instance_sets/luby_train.csv",
    }
)


class CMAESBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for CMA-ES
    """

    def __init__(self, config_path=None):
        super(CMAESEnvBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = CMAES_DEFAULTS

        for key in CMAES_DEFAULTS:
            if not key in self.config:
                self.config[key] = CMAES_DEFAULTS[key]

        if not "instance_set" in self.config.keys():
            self.read_instance_set()

    def get_benchmark_env(self):
        """
        Return Luby env with current configuration

        Returns
        -------
        LubyEnv
            Luby environment
        """
        return CMAESEnv(self.config)

    # TODO: implement instance loading
    def read_instance_set(self):
        pass
