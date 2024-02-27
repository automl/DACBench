"""Luby Benchmark."""
from __future__ import annotations

import csv
from pathlib import Path

import ConfigSpace as CS  # noqa: N817
import ConfigSpace.hyperparameters as CSH
import numpy as np

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import LubyEnv, luby_gen
from dacbench.wrappers import RewardNoiseWrapper

MAX_STEPS = 2**6
LUBY_SEQUENCE = np.log2([next(luby_gen(i)) for i in range(1, 2 * MAX_STEPS + 2)])
HISTORY_LENGTH = 5

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
SEQ = CSH.UniformIntegerHyperparameter(
    name="sequence_element", lower=0, upper=np.log2(MAX_STEPS)
)
DEFAULT_CFG_SPACE.add_hyperparameter(SEQ)

INFO = {
    "identifier": "Luby",
    "name": "Luby Sequence Approximation",
    "reward": "Boolean sucess indication",
    "state_description": [
        "Action t-2",
        "Step t-2",
        "Action t-1",
        "Step t-1",
        "Action t (current)",
        "Step t (current)",
    ],
}

LUBY_DEFAULTS = objdict(
    {
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [
            np.array([-1 for _ in range(HISTORY_LENGTH + 1)]),
            np.array([2 ** max(LUBY_SEQUENCE + 1) for _ in range(HISTORY_LENGTH + 1)]),
        ],
        "reward_range": (-1, 0),
        "cutoff": MAX_STEPS,
        "hist_length": HISTORY_LENGTH,
        "min_steps": 2**3,
        "seed": 0,
        "instance_set_path": "../instance_sets/luby/luby_default.csv",
        "benchmark_info": INFO,
    }
)


class LubyBenchmark(AbstractBenchmark):
    """Benchmark with default configuration & relevant functions for Sigmoid."""

    def __init__(self, config_path=None, config=None):
        """Initialize Luby Benchmark.

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super().__init__(config_path, config)
        if not self.config:
            self.config = objdict(LUBY_DEFAULTS.copy())

        for key in LUBY_DEFAULTS:
            if key not in self.config:
                self.config[key] = LUBY_DEFAULTS[key]

    def get_environment(self):
        """Return Luby env with current configuration.

        Returns:
        -------
        LubyEnv
            Luby environment
        """
        if "instance_set" not in self.config:
            self.read_instance_set()

        # Read test set if path is specified
        if "test_set" not in self.config and "test_set_path" in self.config:
            self.read_instance_set(test=True)

        env = LubyEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def set_cutoff(self, steps):
        """Set cutoff and adapt dependencies.

        Parameters
        -------
        int
            Maximum number of steps
        """
        self.config.cutoff = steps
        self.config.action_space_args = [int(np.log2(steps))]
        LUBY_SEQUENCE = np.log2([next(luby_gen(i)) for i in range(1, 2 * steps + 2)])
        self.config.observation_space_args = [
            np.array([-1 for _ in range(self.config.hist_length + 1)]),
            np.array(
                [
                    2 ** max(LUBY_SEQUENCE + 1)
                    for _ in range(self.config.hist_length + 1)
                ]
            ),
        ]

    def set_history_length(self, length):
        """Set history length and adapt dependencies.

        Parameters
        -------
        int
            History length
        """
        self.config.hist_length = length
        self.config.observation_space_args = [
            np.array([-1 for _ in range(length + 1)]),
            np.array([2 ** max(LUBY_SEQUENCE + 1) for _ in range(length + 1)]),
        ]

    def read_instance_set(self, test=False):
        """Read instance set from file."""
        if test:
            path = Path(__file__).resolve().parent / self.config.test_set_path
            keyword = "test_set"
        else:
            path = Path(__file__).resolve().parent / self.config.instance_set_path
            keyword = "instance_set"

        self.config[keyword] = {}
        with open(path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self.config[keyword][int(row["ID"])] = [
                    float(shift) for shift in row["start"].split(",")
                ] + [float(slope) for slope in row["sticky"].split(",")]

    def get_benchmark(self, min_l=8, fuzziness=1.5, seed=0):
        """Get Benchmark from DAC paper.

        Parameters
        -------
        min_l : int
            Minimum sequence lenght, was 8, 16 or 32 in the paper
        fuzziness : float
            Amount of noise applied. Was 1.5 for most of the experiments
        seed : int
            Environment seed

        Returns:
        -------
        env : LubyEnv
            Luby environment
        """
        self.config = objdict(LUBY_DEFAULTS.copy())
        self.config.min_steps = min_l
        self.config.seed = seed
        self.config.instance_set = {0: [0, 0]}
        self.config.reward_range = (-10, 10)
        env = LubyEnv(self.config)
        rng = np.random.RandomState(self.config.seed)

        def fuzz():
            return rng.normal(-1, fuzziness)

        return RewardNoiseWrapper(env, noise_function=fuzz)
