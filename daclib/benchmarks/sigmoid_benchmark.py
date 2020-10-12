from daclib.abstract_benchmark import AbstractBenchmark, objdict
from daclib.envs import SigmoidEnv
from daclib.wrappers import InstanceSamplingWrapper

import numpy as np
import os
import csv

ACTION_VALUES = (5, 10)

SIGMOID_DEFAULTS = objdict(
    {
        "action_space_class": "Discrete",
        "action_space_args": [int(np.prod(ACTION_VALUES))],
        "observation_space": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [
            np.array([-np.inf for _ in range(1 + len(ACTION_VALUES) * 3)]),
            np.array([np.inf for _ in range(1 + len(ACTION_VALUES) * 3)]),
        ],
        "reward_range": (0, 1),
        "cutoff": 10,
        "action_values": ACTION_VALUES,
        "slope_multiplier": 2.0,
        "seed": 0,
        "instance_set_path": "../instance_sets/sigmoid_train.csv",
    }
)


class SigmoidBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for Sigmoid
    """

    def __init__(self, config_path=None):
        super(SigmoidBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = SIGMOID_DEFAULTS

        for key in SIGMOID_DEFAULTS:
            if key not in self.config:
                self.config[key] = SIGMOID_DEFAULTS[key]

        if "instance_set" not in self.config.keys():
            self.read_instance_set()

    def get_benchmark_env(self):
        """
        Return Sigmoid env with current configuration

        Returns
        -------
        SigmoidEnv
            Sigmoid environment

        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        return SigmoidEnv(self.config)

    def set_action_values(self, values):
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
                self.config.instance_set[int(row["ID"])] = [
                    float(shift) for shift in row["shift"].split(",")
                ] + [float(slope) for slope in row["slope"].split(",")]

    def get_complete_benchmark(self, dimension=None):
        """Get Benchmark from DAC paper"""
        self.config = SIGMOID_DEFAULTS
        if dimension == 1:
            self.set_action_values((3))
        if dimension == 2:
            self.set_action_values((3, 3))
        if dimension == 3:
            self.set_action_values((3, 3, 3))
        if dimension == 5:
            self.set_action_values((3, 3, 3, 3, 3))
        env = SigmoidEnv(self.config)

        def sample_sigmoid():
            shifts = self.rng.normal(
                self.config.cutoff / 2,
                self.config.cutoff / 4,
                self.config.action_space_args[0],
            )
            slopes = (
                self.rng.choice([-1, 1], self.config.action_space_args[0])
                * self.rng.uniform(size=self.config.action_space_args[0])
                * self.config.slope_multiplier
            )
            return np.concatenate((shifts, slopes))

        sampling_env = InstanceSamplingWrapper(env, sampling_function=sample_sigmoid)
        return sampling_env
