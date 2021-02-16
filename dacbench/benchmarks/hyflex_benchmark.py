from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import HyFlexEnv
from gym import spaces
import numpy as np
import os
import csv

HISTORY_LENGTH = 40
INPUT_DIM = 10

INFO = {"identifier": "hyflex",
        "name": "Selection hyper-heuristic",
        "reward": "???",  # TODO
        "state_description": ["Loc",  # TODO
                              "Past Deltas",
                              "Population Size",
                              "Sigma",
                              "History Deltas",
                              "Past Sigma Deltas"]}

HYFLEX_DEFAULTS = objdict(
    {
        "action_space_class": "Box",
        "action_space_args": [np.array([0]), np.array([10])],  # TODO
        "observation_space_class": "Dict",  # TODO
        "observation_space_type": None,  # TODO
        "observation_space_args": [  # TODO
            {
                "current_loc": spaces.Box(
                    low=-np.inf, high=np.inf, shape=np.arange(INPUT_DIM).shape
                ),
                "past_deltas": spaces.Box(
                    low=-np.inf, high=np.inf, shape=np.arange(HISTORY_LENGTH).shape
                ),
                "current_ps": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                "current_sigma": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                "history_deltas": spaces.Box(
                    low=-np.inf, high=np.inf, shape=np.arange(HISTORY_LENGTH * 2).shape
                ),
                "past_sigma_deltas": spaces.Box(
                    low=-np.inf, high=np.inf, shape=np.arange(HISTORY_LENGTH).shape
                ),
            }
        ],
        "reward_range": (-(10 ** 9), 0),  # TODO
        "cutoff": 1e6,
        "seed": 42,
        "instance_set_path": "../instance_sets/hyflex/chesc.csv",
        "benchmark_info": INFO
    }
)


class HyFlexBenchmark(AbstractBenchmark):
    """
    Benchmark for learning a selection hyper-heuristic for HyFlex
    """

    def __init__(self, config_path=None):
        """
        Initialize CMA Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(HyFlexBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = objdict(HYFLEX_DEFAULTS.copy())

        for key in HYFLEX_DEFAULTS:
            if key not in self.config:
                self.config[key] = HYFLEX_DEFAULTS[key]

    def get_environment(self):
        """
        Return HyFlexEnv env with current configuration

        Returns
        -------
        HyFlexEnv
            HyFlex environment
        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        env = HyFlexEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self):
        """
        Read path of instances from config into list
        """
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/"
            + self.config.instance_set_path
        )
        self.config["instance_set"] = {}
        with open(path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                instance = [
                    row["domain"],
                    int(row["instance"]),
                ]
                self.config["instance_set"][int(row["ID"])] = instance
