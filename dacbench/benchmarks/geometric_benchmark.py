from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import GeometricEnv

import numpy as np
import os
import csv

FILE_PATH = os.path.dirname(__file__)


INFO = {
    "identifier": "Geometric",
    "name": "High Dimensional Geometric Curve Approximation",
    "reward": "Multiplied Differences between Function and Action in each Dimension",
    "state_description": [
        "Remaining Budget",
        "Derivative",
        "Trajectories",
        "Coordiantes",
        "Action",
    ],
}

GEOMETRIC_DEFAULTS = objdict(
    {
        "action_space_class": "Discrete",
        "action_space_args": [],
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [],
        "reward_range": (0, 1),
        "cutoff": 10,
        "action_values": [],
        "seed": 0,
        "variable_action_values": True,
        "default_action_value": 4,
        "instance_set_path": "../instance_sets/geometric/geometric_train.csv",
        "benchmark_info": INFO,
    }
)

# apply if "variable_action_value" is set to True
ACTION_VALUE_MAPPING = objdict(
    {
        "sigmoid": 3,
        "linear": 3,
        "polynomial2D": 5,
        "polynomial3D": 7,
        "polynomial7D": 11,
        "exponential": 4,
        "logarithmic": 4,
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
        """
        Read instance set from file
        Creates a nested List for every Intance.
        The List contains all functions with their respective values.
        """
        path = os.path.join(FILE_PATH, self.config.instance_set_path)
        self.config["instance_set"] = {}
        with open(path, "r") as fh:

            known_ids = []
            reader = csv.DictReader(fh)

            for row in reader:
                function_list = []

                if row["ID"] not in known_ids:
                    self.config.instance_set[row["ID"]] = []
                    known_ids.append(row["ID"])

                for index, element in enumerate(row.values()):

                    if element == "0" and index != 0:
                        break

                    # read numbers from csv as floats
                    element = float(element) if index != 1 else element

                    function_list.append(element)

                self.config.instance_set[row["ID"]].append(function_list)

    def get_benchmark(self, dimension=None, seed=0):
        """
        [summary]

        Parameters
        ----------
        dimension : [type], optional
            [description], by default None
        seed : int, optional
            [description], by default 0

        Returns
        -------
        [type]
            [description]
        """
        self.config = objdict(GEOMETRIC_DEFAULTS.copy())

        # Call generator from here, if data not available

        self.config.instance_set_path = "../instance_sets/geometric/geometric_train.csv"
        self.config.benchmark_info["state_description"] = [
            "Remaining Budget",
            "Derivative",
            "Trajectories",
            "Coordinates",
            "Action",
        ]

        self.config.seed = seed
        self.read_instance_set()

        self._set_action_values()

        env = GeometricEnv(self.config)
        return env

    def _set_action_values(self):
        """
        Adapt action values and update dependencies
        Number of actions can differ between functions if configured in DefaultDict
        """
        # create mapping for discretization of each function type
        map_action_number = {}
        if self.config.variable_action_values:
            map_action_number = ACTION_VALUE_MAPPING

        # set action values based on function layers
        values = []
        for function_info in self.config.instance_set["0"]:
            values.append(
                map_action_number.get(
                    function_info[1], self.config.default_action_value
                )
            )

        self.config.action_values = values
        self.config.action_space_args = [int(np.prod(values))]

        # TODO check if observation space is correct in dimension and value
        self.config.observation_space_args = [
            np.array([-1 for _ in range(1 + np.sum(values))]),
            np.array([self.config["cutoff"] for _ in range(1 + np.sum(values))]),
        ]


if __name__ == "__main__":
    geo_env = GeometricBenchmark()
    geo_env.get_benchmark()