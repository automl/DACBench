from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import ModeaEnv
import numpy as np
import os
import csv

INFO = {
    "identifier": "ModEA",
    "name": "Online Selection of CMA-ES Variants",
    "reward": "Negative best function value",
    "state_description": [
        "Generation Size",
        "Sigma",
        "Remaining Budget",
        "Function ID",
        "Instance ID",
    ],
}

MODEA_DEFAULTS = objdict(
    {
        "action_space_class": "MultiDiscrete",
        "action_space_args": [[2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3]],
        "observation_space_class": "Box",
        "observation_space_args": [-np.inf * np.ones(5), np.inf * np.ones(5)],
        "observation_space_type": np.float32,
        "reward_range": (-(10 ** 12), 0),
        "budget": 100,
        "cutoff": 1e6,
        "seed": 0,
        "instance_set_path": "../instance_sets/modea/modea_train.csv",
        "benchmark_info": INFO,
    }
)


class ModeaBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for Modea
    """

    def __init__(self, config_path=None, config=None):
        """
        Initialize Modea Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(ModeaBenchmark, self).__init__(config_path, config)
        if not self.config:
            self.config = objdict(MODEA_DEFAULTS.copy())

        for key in MODEA_DEFAULTS:
            if key not in self.config:
                self.config[key] = MODEA_DEFAULTS[key]

    def get_environment(self):
        """
        Return ModeaEnv env with current configuration

        Returns
        -------
        ModeaEnv
            Modea environment
        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        # Read test set if path is specified
        if "test_set" not in self.config.keys() and "test_set_path" in self.config.keys():
            self.read_instance_set(test=True)

        env = ModeaEnv(self.config)
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
            reader = csv.DictReader(fh)
            for row in reader:
                function = int(row["fcn_id"])
                instance = int(row["inst_id"])
                dimension = int(row["dim"])
                representation = [float(row[f"rep{i}"]) for i in range(11)]
                instance = [
                    dimension,
                    function,
                    instance,
                    representation,
                ]
                self.config[keyword][int(row["ID"])] = instance
