from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import CMAESEnv
from cma import bbobbenchmarks as bn
from gym import spaces
import numpy as np
import os
import csv

HISTORY_LENGTH = 40
INPUT_DIM = 10

INFO = {"identifier": "CMA-ES",
        "name": "Step-size adaption in CMA-ES",
        "reward": "Negative best function value",
        "state_description": ["Loc",
                              "Past Deltas",
                              "Population Size",
                              "Sigma",
                              "History Deltas",
                              "Past Sigma Deltas"]}

CMAES_DEFAULTS = objdict(
    {
        "action_space_class": "Box",
        "action_space_args": [np.array([0]), np.array([10])],
        "observation_space_class": "Dict",
        "observation_space_type": None,
        "observation_space_args": [
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
        "reward_range": (-(10 ** 9), 0),
        "cutoff": 1e6,
        "hist_length": HISTORY_LENGTH,
        "popsize": 10,
        "seed": 0,
        "instance_set_path": "../instance_sets/cma/cma_train.csv",
        "benchmark_info": INFO
    }
)


class CMAESBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for CMA-ES
    """

    def __init__(self, config_path=None):
        """
        Initialize CMA Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(CMAESBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = objdict(CMAES_DEFAULTS.copy())

        for key in CMAES_DEFAULTS:
            if key not in self.config:
                self.config[key] = CMAES_DEFAULTS[key]

    def get_environment(self):
        """
        Return CMAESEnv env with current configuration

        Returns
        -------
        CMAESEnv
            CMAES environment
        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        env = CMAESEnv(self.config)
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
                function = bn.instantiate(int(row["fcn_index"]))[0]
                init_locs = [float(row[f"init_loc{i}"]) for i in range(int(row["dim"]))]
                instance = [
                    function,
                    int(row["dim"]),
                    float(row["init_sigma"]),
                    init_locs,
                ]
                self.config["instance_set"][int(row["ID"])] = instance

    def get_benchmark(self, seed=0):
        """
        Get benchmark from the LTO paper

        Parameters
        -------
        seed : int
            Environment seed

        Returns
        -------
        env : CMAESEnv
            CMAES environment
        """
        self.config = objdict(CMAES_DEFAULTS.copy())
        self.config.seed = seed
        self.read_instance_set()
        return CMAESEnv(self.config)
