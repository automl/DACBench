from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import HyFlexEnv
from gym import spaces
import numpy as np
import sys
import os
import csv


HISTORY_LENGTH = 40
INPUT_DIM = 10

INFO = {"identifier": "hyflex",
        "name": "Selection hyper-heuristic",
        "reward": "Improvement on Best Fitness (f_prev_best - f_new_best)",
        "state_description": ["Fitness Delta (f_incumbent - f_proposed)"]}

HYFLEX_DEFAULTS = objdict(
    {
        "action_space_class": "Discrete",
        "action_space_args": [2],
        "observation_space_class": "Dict",
        "observation_space_type": None,
        "observation_space_args": [
            {
                "f_delta": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,)
                )
            }
        ],
        "reward_range": (0, sys.float_info.max),
        "cutoff": 1e3,
        "seed": 42,
        "instance_set_path": "../instance_sets/hyflex/chesc.csv",
        "benchmark_info": INFO
    }
)


class HyFlexBenchmark(AbstractBenchmark):
    """
    Benchmark for learning a selection hyper-heuristic for HyFlex
    """

    def __init__(self, config_path=None, **kwargs):
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

        for key in kwargs:
            assert(key in HYFLEX_DEFAULTS)
            self.config[key] = kwargs[key]

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

    def get_benchmark(self, seed=0):
        self.config.seed = seed
        return self.get_environment()
    
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
                    int(row["seed"]) if "seed" in row else self.config["seed"],
                    int(row["cutoff"]) if "cutoff" in row else self.config["cutoff"],
                ]
                self.config["instance_set"][int(row["ID"])] = instance
