"""Function Approximation Benchmark."""

from __future__ import annotations

from pathlib import Path

import ConfigSpace as CS  # noqa: N817
import ConfigSpace.hyperparameters as CSH
import numpy as np
import pandas as pd

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import FunctionApproximationEnv, FunctionApproximationInstance
from dacbench.envs.env_utils.toy_functions import get_toy_function

# This configures two action dimensions (functions):
# - The first function should be approximated in a continuous manner
# - The second function should be approximated in a discrete manner
#   for which 10 actions are available
DISCRETE = (False, 10)
DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
DIM1 = CSH.UniformFloatHyperparameter(name="value_dim_1", lower=0, upper=1)
DIM2 = CSH.UniformIntegerHyperparameter(name="value_dim_2", lower=0, upper=10)
DEFAULT_CFG_SPACE.add(DIM1)
DEFAULT_CFG_SPACE.add(DIM2)

INFO = {
    "identifier": "FunctionApproximation",
    "name": "Function Approximation",
    "reward": "Multiplied Differences between Function and Action in each Dimension",
    "state_description": [
        "Remaining Budget",
        "Function Identifier (dimension 1)",
        "Function Parameter 1 (dimension 1)",
        "Function Parameter 2 (dimension 1)",
        "Function Identifier (dimension 2)",
        "Function Parameter 1 (dimension 2)",
        "Function Parameter 2 (dimension 2)",
        "Action 1",
        "Action 2",
    ],
}

FUNCTION_APPROXIMATION_DEFAULTS = objdict(
    {
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [
            np.array([-np.inf for _ in range(1 + len(DEFAULT_CFG_SPACE) * 4)]),
            np.array([np.inf for _ in range(1 + len(DEFAULT_CFG_SPACE) * 4)]),
        ],
        "discrete": DISCRETE,
        "reward_range": (0, 1),
        "cutoff": 10,
        "seed": 0,
        "multi_agent": False,
        "omit_instance_type": False,
        "instance_set_path": "sigmoid_2D3M_train.csv",
        "test_set_path": "sigmoid_2D3M_test.csv",
        "benchmark_info": INFO,
    }
)


class FunctionApproximationBenchmark(AbstractBenchmark):
    """Benchmark with default configuration &
    relevant functions for Function Approximation.
    """

    @staticmethod
    def _isolate_benchmark_info(config):
        """Deep-copy `benchmark_info` (and its state_description list) so
        that per-instance mutations do not leak into the module-level INFO
        dict. Idempotent: safe to call multiple times on the same config.
        """
        config["benchmark_info"] = objdict(config["benchmark_info"])
        config["benchmark_info"]["state_description"] = list(
            config["benchmark_info"]["state_description"]
        )

    def __init__(self, config_path=None, config=None):
        """Initialize Function Approximation Benchmark.

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super().__init__(config_path, config)
        if not self.config:
            self.config = objdict(FUNCTION_APPROXIMATION_DEFAULTS.copy())

        for key in FUNCTION_APPROXIMATION_DEFAULTS:
            if key not in self.config and key != "observation_space_args":
                self.config[key] = FUNCTION_APPROXIMATION_DEFAULTS[key]

        self._isolate_benchmark_info(self.config)

        if "observation_space_args" not in self.config:
            if "config_space" in self.config:
                # Derive obs length from `state_description` so it stays
                # consistent regardless of `omit_instance_type` or the
                # length of `instance_description` returned by the toy
                # functions. The state produced by FunctionApproximationEnv
                # has exactly one entry per item in `state_description`.
                # Deferred until after the default-fill loop so that
                # `benchmark_info.state_description` is guaranteed to be set.
                obs_length = len(self.config["benchmark_info"]["state_description"])
                self.config["observation_space_args"] = [
                    np.array([-np.inf for _ in range(obs_length)]),
                    np.array([np.inf for _ in range(obs_length)]),
                ]
            else:
                self.config["observation_space_args"] = FUNCTION_APPROXIMATION_DEFAULTS[
                    "observation_space_args"
                ]

    def get_environment(self):
        """Return Function Approximation env with current configuration.

        Returns:
        -------
        FunctionApproximationEnv
            Function Approximation environment

        """
        if "instance_set" not in self.config:
            self.read_instance_set()

        # Read test set if path is specified
        if "test_set" not in self.config and "test_set_path" in self.config:
            self.read_instance_set(test=True)

        env = FunctionApproximationEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self, test=False):
        """Read instance set from file."""
        if test:
            path = Path(self.config.test_set_path)
            relative_path = Path(__file__).resolve().parent / self.config.test_set_path
            dacbench_path = (
                Path(__file__).resolve().parent
                / "../instance_sets/function_approximation"
                / self.config.test_set_path
            )
            keyword = "test_set"
        else:
            path = Path(self.config.instance_set_path)
            relative_path = (
                Path(__file__).resolve().parent / self.config.instance_set_path
            )
            dacbench_path = (
                Path(__file__).resolve().parent
                / "../instance_sets/function_approximation"
                / self.config.instance_set_path
            )
            keyword = "instance_set"

        if path.is_file():
            path = path  # noqa: PLW0127
        elif relative_path.is_file():
            path = relative_path
        elif dacbench_path.is_file():
            path = dacbench_path
        else:
            raise FileNotFoundError(
                f"Test set not found at {self.config.test_set_path}"
            )

        self.config[keyword] = {}
        instances = pd.read_csv(path)
        num_functions = (len(instances.columns) - 1) // 3
        for i, row in instances.iterrows():
            functions = []
            for j in range(num_functions):
                functions.append(
                    get_toy_function(
                        identifier=row[f"function_{j}_identifier"],
                        a=row[f"function_{j}_a"],
                        b=row[f"function_{j}_b"],
                    )
                )
            importances = [1 / num_functions for _ in range(num_functions)]
            if "importance_dim_0" in row:
                importances = [row[f"importance_dim_{j}"] for j in range(num_functions)]

            self.config[keyword][i] = FunctionApproximationInstance(
                functions=functions,
                dimension_importances=importances,
                discrete=self.config.discrete,
                omit_instance_type=self.config.omit_instance_type,
            )

    def get_benchmark(self, dimension=None, seed=0):
        """Get Sigmoid Benchmark from DAC paper.

        Parameters
        -------
        dimension : int
            Sigmoid dimension, was 1, 2, 3 or 5 in the paper
        seed : int
            Environment seed

        Returns:
        -------
        env : SigmoidEnv
            Sigmoid environment
        """
        self.config = objdict(FUNCTION_APPROXIMATION_DEFAULTS.copy())
        self._isolate_benchmark_info(self.config)
        self.config.omit_instance_type = True
        if dimension == 1:
            self.config.instance_set_path = "sigmoid_1D3M_train.csv"
            self.config.test_set_path = "sigmoid_1D3M_test.csv"
            self.config.discrete = [3]
            cfg_space = CS.ConfigurationSpace()
            dim1 = CSH.UniformIntegerHyperparameter(
                name="value_dim_1", lower=0, upper=2
            )
            cfg_space.add(dim1)
            self.config.config_space = cfg_space
            self.config.benchmark_info["state_description"] = [
                "Remaining Budget",
                "Shift (dimension 1)",
                "Slope (dimension 1)",
                "Action",
            ]
            self.config.observation_space_args = [
                np.array([-np.inf for _ in range(4)]),
                np.array([np.inf for _ in range(4)]),
            ]
        if dimension == 2:
            self.config.instance_set_path = "sigmoid_2D3M_train.csv"
            self.config.test_set_path = "sigmoid_2D3M_test.csv"
            self.config.discrete = [3, 3]
            cfg_space = CS.ConfigurationSpace()
            dim1 = CSH.UniformIntegerHyperparameter(
                name="value_dim_1", lower=0, upper=2
            )
            dim2 = CSH.UniformIntegerHyperparameter(
                name="value_dim_2", lower=0, upper=2
            )
            cfg_space.add(dim1)
            cfg_space.add(dim2)
            self.config.config_space = cfg_space
            self.config.benchmark_info["state_description"] = [
                "Remaining Budget",
                "Shift (dimension 1)",
                "Slope (dimension 1)",
                "Shift (dimension 2)",
                "Slope (dimension 2)",
                "Action dim 1",
                "Action dim 2",
            ]
            self.config.observation_space_args = [
                np.array([-np.inf for _ in range(7)]),
                np.array([np.inf for _ in range(7)]),
            ]
        if dimension == 3:
            self.config.instance_set_path = "sigmoid_3D3M_train.csv"
            self.config.test_set_path = "sigmoid_3D3M_test.csv"
            self.config.discrete = [3, 3, 3]
            cfg_space = CS.ConfigurationSpace()
            dim1 = CSH.UniformIntegerHyperparameter(
                name="value_dim_1", lower=0, upper=2
            )
            dim2 = CSH.UniformIntegerHyperparameter(
                name="value_dim_2", lower=0, upper=2
            )
            dim3 = CSH.UniformIntegerHyperparameter(
                name="value_dim_3", lower=0, upper=2
            )
            cfg_space.add(dim1)
            cfg_space.add(dim2)
            cfg_space.add(dim3)
            self.config.config_space = cfg_space
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
            self.config.observation_space_args = [
                np.array([-np.inf for _ in range(10)]),
                np.array([np.inf for _ in range(10)]),
            ]
        if dimension == 5:
            self.config.instance_set_path = "sigmoid_5D3M_train.csv"
            self.config.test_set_path = "sigmoid_5D3M_test.csv"
            self.config.discrete = [3, 3, 3, 3, 3]
            cfg_space = CS.ConfigurationSpace()
            dim1 = CSH.UniformIntegerHyperparameter(
                name="value_dim_1", lower=0, upper=2
            )
            dim2 = CSH.UniformIntegerHyperparameter(
                name="value_dim_2", lower=0, upper=2
            )
            dim3 = CSH.UniformIntegerHyperparameter(
                name="value_dim_3", lower=0, upper=2
            )
            dim4 = CSH.UniformIntegerHyperparameter(
                name="value_dim_4", lower=0, upper=2
            )
            dim5 = CSH.UniformIntegerHyperparameter(
                name="value_dim_5", lower=0, upper=2
            )
            cfg_space.add(dim1)
            cfg_space.add(dim2)
            cfg_space.add(dim3)
            cfg_space.add(dim4)
            cfg_space.add(dim5)
            self.config.config_space = cfg_space
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
            self.config.observation_space_args = [
                np.array([-np.inf for _ in range(16)]),
                np.array([np.inf for _ in range(16)]),
            ]
        self.config.seed = seed
        self.read_instance_set()
        self.read_instance_set(test=True)
        return FunctionApproximationEnv(self.config)
