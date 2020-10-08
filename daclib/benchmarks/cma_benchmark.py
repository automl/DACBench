from daclib.abstract_benchmark import AbstractBenchmark, objdict

from daclib.envs import CMAESEnv
from daclib.benchmarks.cma_fcn import FcnFamily

from gym import spaces
import numpy as np
import os
import csv

HISTORY_LENGTH = 5
MAX_DIM = 10

# TODO: fix this
CMAES_DEFAULTS = objdict(
    {
        "action_space": "Box",
        "action_space_args": [-np.inf*np.ones(MAX_DIM), np.inf*np.ones(MAX_DIM)],
        "observation_space": "Dict",
        "observation_space_type": None,
        "observation_space_args": {
            "past_deltas": spaces.Box(low=-np.inf, high=np.inf, shape=np.arange(HISTORY_LENGTH).shape),
            "current_ps": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            "current_sigma": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            "history_deltas": spaces.Box(
                low=-np.inf, high=np.inf, shape=np.arange(HISTORY_LENGTH * 2).shape
            ),
            "past_sigma": spaces.Box(low=-np.inf, high=np.inf, shape=np.arange(HISTORY_LENGTH).shape),
        },
        "reward_range": (-np.inf, np.inf),
        "cutoff": 1e6,
        "hist_length": HISTORY_LENGTH,
        "popsize": 1,
        "seed": 0,
        "instance_set_path": "../instance_sets/cma_train.csv",
    }
)


class CMAESBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for CMA-ES
    """

    def __init__(self, config_path=None):
        super(CMAESBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = CMAES_DEFAULTS

        for key in CMAES_DEFAULTS:
            if not key in self.config:
                self.config[key] = CMAES_DEFAULTS[key]

    def get_benchmark_env(self):
        """
        Return CMAESEnv env with current configuration

        Returns
        -------
        LubyEnv
            Luby environment
        """
        if not "instance_set" in self.config.keys():
            self.read_instance_set()
        return CMAESEnv(self.config)

    # TODO: test this
    def read_instance_set(self):
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/"
            + self.config.instance_set_path
        )
        self.config["instance_set"] = {}
        with open(path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                instance = (
                    [float(loc) for loc in row["loc"].split(",")]
                    + [float(sigma) for sigma in row["sigma"].split(",")]
                    + [float(popsize) for popsize in row["popsize"].split(",")]
                    + [int(dim) for dim in row["dim"].split(",")]
                )
                if "func" in row.keys():
                    func_args = [float(arg) for arg in row["args"].split(",")]
                    instance.append(getattr(FcnFamiliy, row["func"])(*func_args))
                self.config["instance_set"][int(row["ID"])] = instance
