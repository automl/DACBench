"""Geometric Benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

import ConfigSpace as CS  # noqa: N817
import ConfigSpace.hyperparameters as CSH
import numpy as np

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import GeometricEnv

FILE_PATH = Path(__file__).resolve().parent
ACTION_VALUES = (5, 10)

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()

INFO = {
    "identifier": "Geometric",
    "name": (
        "High Dimensional Geometric Curve Approximation. "
        "Curves are geometrical orthogonal."
    ),
    "reward": (
        "Overall Euclidean Distance between Point on Curve "
        "and Action Vector for all Dimensions"
    ),
    "state_description": [
        "Remaining Budget",
        "Dimensions",
    ],
}

GEOMETRIC_DEFAULTS = objdict(
    {
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [],
        "reward_range": (0, 1),
        "seed": 0,
        "multi_agent": False,
        "cutoff": 10,
        "action_values": [],
        "action_value_default": 4,
        # if action_values_variable True action_value_mapping will be used instead of
        # action_value_default to define action values
        # action_value_mapping defines number of action values for differnet functions
        # sigmoid is split in 3 actions, cubic in 7 etc.
        "action_values_variable": False,
        "action_value_mapping": {
            "sigmoid": 3,
            "linear": 3,
            "parabel": 5,
            "cubic": 7,
            "logarithmic": 4,
            "constant": 1,
            "sinus": 9,
        },
        "action_interval_mapping": {},  # maps actions to equally sized intervalls
        # in interval [-1, 1]
        "derivative_interval": 3,  # defines how many values are used for
        # derivative calculation
        "realistic_trajectory": True,  # True: coordiantes are used as trajectory,
        # False: Actions are used as trajectories
        "instance_set_path": Path(FILE_PATH)
        / "../instance_sets/geometric/geometric_test.csv",
        # correlation table to chain dimensions-> if dim x changes dim y changes as well
        # either assign numpy array to correlation
        # table or use create_correlation_table()
        "correlation_active": False,
        "correlation_table": None,
        "correlation_info": {
            "high": [(1, 2, "+"), (2, 3, "-"), (1, 5, "+")],
            "middle": [(4, 5, "-")],
            "low": [(4, 6, "+"), (2, 3, "+"), (0, 2, "-")],
        },
        "correlation_mapping": {
            "high": (0.5, 1),
            "middle": (0.1, 0.5),
            "low": (0, 0.1),
        },
        "correlation_depth": 4,
        "benchmark_info": INFO,
    }
)


class GeometricBenchmark(AbstractBenchmark):
    """Benchmark with default configuration & relevant functions for Geometric."""

    def __init__(self, config_path=None):
        """Initialize Geometric Benchmark.

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super().__init__(config_path)
        if not self.config:
            self.config = objdict(GEOMETRIC_DEFAULTS.copy())

        for key in GEOMETRIC_DEFAULTS:
            if key not in self.config:
                self.config[key] = GEOMETRIC_DEFAULTS[key]

        if not self.config["observation_space_type"]:
            self.config["observation_space_type"] = np.float32

    def get_environment(self):
        """Return Geometric env with current configuration.

        Returns:
        -------
        GeometricEnv
            Geometric environment

        """
        if "instance_set" not in self.config:
            self.read_instance_set()

        self.set_action_values()
        self.set_action_description()

        if self.config.correlation_active and not isinstance(
            self.config.correlation_table, np.ndarray
        ):
            self.create_correlation_table()

        env = GeometricEnv(self.config)

        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self):
        """Read instance set from file
        Creates a nested List for every Intance.
        The List contains all functions with their respective values.
        """
        path = Path(FILE_PATH) / self.config.instance_set_path
        self.config["instance_set"] = {}
        with open(path) as fh:
            known_ids = []
            reader = csv.DictReader(fh)

            for row in reader:
                function_list = []
                row_id = int(row["ID"])

                if row_id not in known_ids:
                    self.config.instance_set[row_id] = []
                    known_ids.append(row_id)

                for index, element in enumerate(row.values()):
                    # if element == "0" and index != 0:
                    #     break

                    # read numbers from csv as floats
                    elem = float(element) if index != 1 else element

                    function_list.append(elem)

                self.config.instance_set[row_id].append(function_list)

    def get_benchmark(self, dimension=None, seed=0):
        """[summary].

        Parameters
        ----------
        dimension : [type], optional
            [description], by default None
        seed : int, optional
            [description], by default 0

        Returns:
        -------
        [type]
            [description]
        """
        self.config = objdict(GEOMETRIC_DEFAULTS.copy())

        self.config.benchmark_info["state_description"] = [
            "Remaining Budget",
            "Dimensions",
        ]

        self.config.seed = seed
        if "instance_set" not in self.config:
            self.read_instance_set()

        self.set_action_values()
        self.set_action_description()

        if self.config.correlation_active and not isinstance(
            self.config.correlation_table, np.ndarray
        ):
            self.create_correlation_table()

        return GeometricEnv(self.config)

    def set_action_values(self):
        """Adapt action values and update dependencies
        Number of actions can differ between functions if configured in DefaultDict
        Set observation space args.
        """
        map_action_number = {}
        if self.config.action_values_variable:
            map_action_number = self.config.action_value_mapping

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

                self.config.action_interval_mapping[function_name] = np.round(
                    action_interval, 3
                )

        self.config.action_values = values
        cs = CS.ConfigurationSpace()
        for i, v in enumerate(values):
            actions = CSH.UniformIntegerHyperparameter(
                name=f"curve_values_dim_{i}", lower=0, upper=v
            )
            cs.add_hyperparameter(actions)
        self.config.config_space = cs

        num_info = 2
        self.config.observation_space_args = [
            np.array([-1 for _ in range(num_info + 2 * len(values))]),
            np.array(
                [self.config["cutoff"] for _ in range(num_info + 2 * len(values))]
            ),
        ]

    def set_action_description(self):
        """Add Information about Derivative and Coordinate to Description."""
        if "Coordinate" in self.config.benchmark_info["state_description"]:
            return

        for index in range(len(self.config.action_values)):
            self.config.benchmark_info["state_description"].append(f"Derivative{index}")

        for index in range(len(self.config.action_values)):
            self.config.benchmark_info["state_description"].append(f"Coordinate{index}")

    def create_correlation_table(self):
        """Create correlation table from Config infos."""
        rng = np.random.default_rng()
        n_dimensions = len(self.config.instance_set[0])
        corr_table = np.zeros((n_dimensions, n_dimensions))

        for corr_level, corr_info in self.config.correlation_info.items():
            for dim1, dim2, signum in corr_info:
                low, high = self.config.correlation_mapping[corr_level]
                value = rng.uniform(low, high)
                try:
                    corr_table[dim1, dim2] = value if signum == "+" else value * -1
                except IndexError:
                    print(
                        "Check your correlation_info dict. Does it have more dimensions"
                        " than the instance_set?"
                    )

        self.config.correlation_table = corr_table


if __name__ == "__main__":
    from dacbench.challenge_benchmarks.reward_quality_challenge.reward_functions import (  # noqa: E501
        quadratic_euclidean_distance_reward_geometric,
    )

    geo_bench = GeometricBenchmark()
    geo_bench.config["correlation_active"] = True
    geo_bench.config["reward_function"] = quadratic_euclidean_distance_reward_geometric

    env = geo_bench.get_environment()

    opt_policy = env.get_optimal_policy()
    # env.render_dimensions([0, 1, 2, 3, 4, 5, 6], "/home/vonglahn/tmp/MultiDAC")
    env.render_3d_dimensions([1, 3], "/home/eimer/tmp")

    rng = np.random.default_rng()
    while True:
        env.reset()
        done = False
        while not done:
            state, reward, done, info = env.step(rng.randint(env.action_space.n))
            print(reward)
