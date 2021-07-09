from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import GeometricEnv

import numpy as np
import os
import csv


ACTION_VALUES = (5, 10)


INFO = {
    "identifier": "Geometric",
    "name": "High Dimensional Geometric Curve Approximation",
    "reward": "Multiplied Differences between Function and Action in each Dimension",
    "state_description": [
        "Remaining Budget",
        "Shift (dimension 1)",
        "Slope (dimension 1)",
        "Shift (dimension 2)",
        "Slope (dimension 2)",
        "Action 1",
        "Action 2",
    ],
}

GEOMETRIC_DEFAULTS = objdict(
    {
        "action_space_class": "Discrete",
        "action_space_args": [[2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3]],
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [
            np.array([-np.inf for _ in range(1 + len(ACTION_VALUES) * 3)]),
            np.array([np.inf for _ in range(1 + len(ACTION_VALUES) * 3)]),
        ],
        "reward_range": (0, 1),
        "cutoff": 10,
        "action_values": ACTION_VALUES,
        "seed": 0,
        "instance_set_path": "../instance_sets/geometric/geometric_train.csv",
        "benchmark_info": INFO,
    }
)


class GeometricBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for Sigmoid
    """

    def __init__(self, config_path=None):
        """
        Initialize Sigmoid Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(GeometricBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = objdict(GEOMETRIC_DEFAULTS.copy())

        for key in GEOMETRIC_DEFAULTS:
            if key not in self.config:
                self.config[key] = GEOMETRIC_DEFAULTS[key]

    def get_environment(self):
        """
        Return Sigmoid env with current configuration

        Returns
        -------
        SigmoidEnv
            Sigmoid environment

        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        env = GeometricEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self):
        """Read instance set from file"""
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/"
            + self.config.instance_set_path
        )
        self.config["instance_set"] = {}
        with open(path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # TODO: - first create instance set for different configurations
                pass

    def get_benchmark(self, dimension=None, seed=0):
        self.config = objdict(GEOMETRIC_DEFAULTS.copy())
        # TODO: wie wird instance_set generiert? Dynamisch statisch? Welche Vorgaben? Generator oder json?

        if dimension == 1:
            self._set_action_values([3])
            self.config.instance_set_path = (
                "../instance_sets/geometric/geometric_XY_train.csv"
            )
            self.config.benchmark_info["state_description"] = [
                "Remaining Budget",
                "tba",
                "Action",
            ]

        self.config.seed = seed
        self.read_instance_set()
        env = GeometricEnv(self.config)
        return env

    def _set_action_values(self, values):
        """
        Adapt action values and update dependencies

        Parameters
        ----------
        values: list
            A list of possible actions per dimension
        """
        self.config.action_values = values
        self.config.action_space_args = [int(np.prod(values))]
        self.config.observation_space_args = [
            np.array([-np.inf for _ in range(1 + len(values) * 3)]),
            np.array([np.inf for _ in range(1 + len(values) * 3)]),
        ]