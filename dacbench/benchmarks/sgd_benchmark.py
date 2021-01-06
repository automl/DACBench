from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import SGDEnv
from gym import spaces
import numpy as np
import os
import csv

HISTORY_LENGTH = 40
INPUT_DIM = 10

INFO = {"identifier": "LR",
        "name": "Learning Rate Adaption for Neural Networks",
        "reward": "Validation Loss",
        "state_description": ["Predictive Change Variance (Discounted Average)",
                              "Predictive Change Variance (Uncertainty)",
                              "Loss Variance (Discounted Average)",
                              "Loss Variance (Uncertainty)",
                              "Current Learning Rate",
                              "Training Loss",
                              "Validation Loss"]}

SGD_DEFAULTS = objdict(
    {
        "action_space_class": "Box",
        "action_space_args": [np.array([0]), np.array([10])],
        "observation_space_class": "Dict",
        "observation_space_type": None,
        "observation_space_args": [
            {
                "predictiveChangeVarDiscountedAverage": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,)
                ),
                "predictiveChangeVarUncertainty": spaces.Box(
                    low=0, high=np.inf, shape=(1,)
                ),
                "lossVarDiscountedAverage": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                "lossVarUncertainty": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "currentLR": spaces.Box(low=0, high=1, shape=(1,)),
                "trainingLoss": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "validationLoss": spaces.Box(low=0, high=np.inf, shape=(1,)),
            }
        ],
        "reward_range": (-(10 ** 9), 0),
        "cutoff": 5e1,
        "lr": 1e-3,
        "training_batch_size": 64,
        "validation_batch_size": 64,
        "no_cuda": False,
        "beta1": 0.9,
        "beta2": 0.999,
        "seed": 0,
        "instance_set_path": "../instance_sets/sgd/sgd_train.csv",
        "benchmark_info": INFO
    }
)


class SGDBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for SGD
    """

    def __init__(self, config_path=None):
        """
        Initialize SGD Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(SGDBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = objdict(SGD_DEFAULTS.copy())

        for key in SGD_DEFAULTS:
            if key not in self.config:
                self.config[key] = SGD_DEFAULTS[key]

    def get_environment(self):
        """
        Return SGDEnv env with current configuration

        Returns
        -------
        SGDEnv
            SGD environment
        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        env = SGDEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self):
        """
        Read path of instances from config into list
        """

        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/"
            + self.config.instance_set_path
        )
        self.config["instance_set"] = {}
        with open(path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                instance = [
                    row["dataset"],
                    int(row["seed"]),
                ]
                self.config["instance_set"][int(row["ID"])] = instance

    def get_benchmark(self, seed=0):
        """
        Get benchmark from the LTO paper

        Parameters
        -------
        seed : int
            Environment seed

        Returns
        -------
        env : SGDEnv
            SGD environment
        """
        self.config = objdict(SGD_DEFAULTS.copy())
        self.config.seed = seed
        self.read_instance_set()
        return SGDEnv(self.config)
