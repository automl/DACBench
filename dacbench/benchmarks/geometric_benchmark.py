from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import GeometricEnv

import numpy as np


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
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "reward_range": (0, 1),
        "cutoff": 10,
        "slope_multiplier": 2.0,
        "seed": 0,
        "instance_set_path": "../instance_sets/sigmoid/sigmoid_2D3M_train.csv",
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

    def set_action_values(self, values):
        """
        Adapt action values and update dependencies

        Parameters
        ----------
        values: list
            A list of possible actions per dimension
        """
        pass

    def read_instance_set(self):
        """Read instance set from file"""
        pass

    def get_benchmark(self, dimension=None, seed=0):
        pass