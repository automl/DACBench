import csv
import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import ContinuousSigmoidEnv, ContinuousStateSigmoidEnv, SigmoidEnv

ACTION_VALUES = (5, 10)

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
for i, d in enumerate(ACTION_VALUES):
    X = CSH.UniformIntegerHyperparameter(name=f"value_dim_{i}", lower=0, upper=d - 1)
    DEFAULT_CFG_SPACE.add_hyperparameter(X)

INFO = {
    "identifier": "Sigmoid",
    "name": "Sigmoid Function Approximation",
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

SIGMOID_DEFAULTS = objdict(
    {
        "config_space": DEFAULT_CFG_SPACE,
        "action_space_class": "MultiDiscrete",
        "action_space_args": [ACTION_VALUES],
        "observation_space_class": "Box",
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
        "multi_agent": False,
        "default_action": [0, 0],
        "instance_set_path": "../instance_sets/sigmoid/sigmoid_2D3M_train.csv",
        "test_set_path": "../instance_sets/sigmoid/sigmoid_2D3M_test.csv",
        "benchmark_info": INFO,
    }
)


class SigmoidBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for Sigmoid
    """

    def __init__(self, config_path=None, config=None):
        """
        Initialize Sigmoid Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(SigmoidBenchmark, self).__init__(config_path, config)
        if not self.config:
            self.config = objdict(SIGMOID_DEFAULTS.copy())

        for key in SIGMOID_DEFAULTS:
            if key not in self.config:
                self.config[key] = SIGMOID_DEFAULTS[key]

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

        # Read test set if path is specified
        if (
            "test_set" not in self.config.keys()
            and "test_set_path" in self.config.keys()
        ):
            self.read_instance_set(test=True)

        if (
            "env_type" in self.config
        ):  # The env_type determines which Sigmoid environment to use.
            if self.config["env_type"].lower() in [
                "continuous",
                "cont",
            ]:  # Either continuous ...
                if (
                    self.config["action_space"] == "Box"
                ):  # ... in both actions and x-axis state, only ...
                    env = ContinuousSigmoidEnv(self.config)
                elif (
                    self.config["action_space"] == "MultiDiscrete"
                ):  # ... continuous in the x-axis state or ...
                    env = ContinuousStateSigmoidEnv(self.config)
                else:
                    raise Exception(
                        f'The given environment type "{self.config["env_type"]}" does not support the'
                        f' chosen action_space "{self.config["action_space"]}". The action space has to'
                        f' be either of type "Box" for continuous actions or "Discrete".'
                    )
            else:  # ... discrete.
                env = SigmoidEnv(self.config)
        else:  # If the type is not specified we the simplest, fully discrete version.
            env = SigmoidEnv(self.config)
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
        del self.config["config_space"]
        self.config.action_space_args = [values]
        self.config.observation_space_args = [
            np.array([-np.inf for _ in range(1 + len(values) * 3)]),
            np.array([np.inf for _ in range(1 + len(values) * 3)]),
        ]

    def read_instance_set(self, test=False):
        """Read instance set from file"""
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
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                f = []
                inst_id = None
                for i in range(len(row)):
                    if i == 0:
                        try:
                            inst_id = int(row[i])
                        except Exception:
                            continue
                    else:
                        try:
                            f.append(float(row[i]))
                        except Exception:
                            continue

                if not len(f) == 0:
                    self.config[keyword][inst_id] = f

    def get_benchmark(self, dimension=None, seed=0):
        """
        Get Benchmark from DAC paper

        Parameters
        -------
        dimension : int
            Sigmoid dimension, was 1, 2, 3 or 5 in the paper
        seed : int
            Environment seed

        Returns
        -------
        env : SigmoidEnv
            Sigmoid environment
        """
        self.config = objdict(SIGMOID_DEFAULTS.copy())
        if dimension == 1:
            self.set_action_values([3])
            self.config.instance_set_path = (
                "../instance_sets/sigmoid/sigmoid_1D3M_train.csv"
            )
            self.config.test_set_path = "../instance_sets/sigmoid/sigmoid_1D3M_test.csv"
            self.config.benchmark_info["state_description"] = [
                "Remaining Budget",
                "Shift (dimension 1)",
                "Slope (dimension 1)",
                "Action",
            ]
        if dimension == 2:
            self.set_action_values([3, 3])
        if dimension == 3:
            self.set_action_values((3, 3, 3))
            self.config.instance_set_path = (
                "../instance_sets/sigmoid/sigmoid_3D3M_train.csv"
            )
            self.config.test_set_path = "../instance_sets/sigmoid/sigmoid_3D3M_test.csv"
            self.config.benchmark_info["state_description"] = [
                "Remaining Budget",
                "Shift (dimension 1)",
                "Slope (dimension 1)",
                "Shift (dimension 2)",
                "Slope (dimension 2)",
                "Shift (dimension 3)",
                "Slope (dimension 3)",
                "Action 1",
                "Action 2",
                "Action 3",
            ]
        if dimension == 5:
            self.set_action_values((3, 3, 3, 3, 3))
            self.config.instance_set_path = (
                "../instance_sets/sigmoid/sigmoid_5D3M_train.csv"
            )
            self.config.test_set_path = "../instance_sets/sigmoid/sigmoid_5D3M_test.csv"
            self.config.benchmark_info["state_description"] = [
                "Remaining Budget",
                "Shift (dimension 1)",
                "Slope (dimension 1)",
                "Shift (dimension 2)",
                "Slope (dimension 2)",
                "Shift (dimension 3)",
                "Slope (dimension 3)",
                "Shift (dimension 4)",
                "Slope (dimension 4)",
                "Shift (dimension 5)",
                "Slope (dimension 5)",
                "Action 1",
                "Action 2",
                "Action 3",
                "Action 4",
                "Action 5",
            ]
        self.config.seed = seed
        self.read_instance_set()
        self.read_instance_set(test=True)
        env = SigmoidEnv(self.config)
        return env
