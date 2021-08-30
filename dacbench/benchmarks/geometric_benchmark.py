from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import GeometricEnv

import numpy as np
import os
import csv

FILE_PATH = os.path.dirname(__file__)
ACTION_VALUES = (5, 10)

INFO = {
    "identifier": "Geometric",
    "name": "High Dimensional Geometric Curve Approximation. Curves are geometrical orthogonal.",
    "reward": "Overall Euclidean Distance between Point on Curve and Action Vector for all Dimensions",
    "state_description": [
        "Remaining Budget",
        "Derivative",
        "Trajectory",
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
        "action_value_default": 4,
        "action_values_variable": False,  # if True action value mapping will be used
        "action_value_mapping": {  # defines number of action values for differnet functions
            "sigmoid": 3,
            "linear": 3,
            "polynomial2D": 5,
            "polynomial3D": 7,
            "polynomial7D": 11,
            "exponential": 4,
            "logarithmic": 4,
            "constant": 1,
        },
        "action_interval_mapping": {},  # maps actions to equally sized intervalls in [-1, 1]
        "seed": 0,
        "derivative_interval": 3,  # defines how many values are used for derivative calculation
        "max_function_value": 10000,  # clip function value if it is higher than this number
        "realistic_trajectory": True,  # True: coordiantes are used as trajectory, False: Actions are used as trajectories
        "instance_set_path": "../instance_sets/geometric/geometric_train.csv",
        "benchmark_info": INFO,
    }
)


class GeometricBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for Geometric
    """

    def __init__(self, config_path=None):
        """
        Initialize Geometric Benchmark

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

        if self.config["observation_space_type"] == None:
            self.config["observation_space_type"] = np.float32

    def get_environment(self):
        """
        Return Geometric env with current configuration

        Returns
        -------
        GeometricEnv
            Geometric environment

        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        self.set_action_values()
        self.set_action_description()

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
                id = int(row["ID"])

                if id not in known_ids:
                    self.config.instance_set[id] = []
                    known_ids.append(id)

                for index, element in enumerate(row.values()):
                    if element == "0" and index != 0:
                        break

                    # read numbers from csv as floats
                    element = float(element) if index != 1 else element

                    function_list.append(element)

                self.config.instance_set[id].append(function_list)

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

        self.config.benchmark_info["state_description"] = [
            "Remaining Budget",
            "Derivative",
            "Trajectory",
            "Action",
        ]

        self.config.seed = seed
        self.read_instance_set()

        self.set_action_values()
        self.set_action_description()

        env = GeometricEnv(self.config)
        return env

    def set_action_values(self):
        """
        Adapt action values and update dependencies
        Number of actions can differ between functions if configured in DefaultDict
        """
        # create mapping for discretization of each function type
        map_action_number = {}
        if self.config.action_values_variable:
            map_action_number = self.config.action_value_mapping

        # set action values based on mapping, if not existent set default
        values = []

        for function_info in self.config.instance_set[0]:
            function_name = function_info[1]

            value = map_action_number.get(
                function_name, self.config.action_value_default
            )
            values.append(value)

            # map intervall [-1, 1] to action values
            if function_name not in self.config.action_interval_mapping:
                action_interval = []
                step_size = 2 / value

                for step in np.arange(-1, 1, step_size):
                    lower_bound = step
                    upper_bound = step + step_size
                    middle = (lower_bound + upper_bound) / 2

                    action_interval.append(middle)

                self.config.action_interval_mapping[function_name] = action_interval

        self.config.action_values = values
        self.config.action_space_args = [int(np.prod(values))]

        self.config.observation_space_args = [
            np.array([-1 for _ in range(1 + np.sum(values))]),
            np.array([self.config["cutoff"] for _ in range(1 + np.sum(values))]),
        ]

    def set_action_description(self):
        for index, _ in enumerate(self.config.action_values):
            self.config.benchmark_info["state_description"].append(
                "Action" + str(index)
            )


if __name__ == "__main__":
    geo_bench = GeometricBenchmark()
    geo_bench.read_instance_set()
    geo_bench.set_action_values()

    config = GEOMETRIC_DEFAULTS
    config["instance_set"] = geo_bench.config.instance_set
    config["action_values"] = geo_bench.config.action_values
    config["action_space_args"] = geo_bench.config.action_space_args
    config["observation_space_args"] = geo_bench.config.observation_space_args

    env = GeometricEnv(config)
    opt_policy = env.get_optimal_policy()
    env.render_dimension([3], "master")
    env.reset()

    for step in range(env.n_steps):
        env.step(3)
