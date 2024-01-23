import csv
import os

import ConfigSpace as CS
import numpy as np
from gymnasium import spaces
from torch import nn

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import SGDEnv
from dacbench.envs.env_utils import utils

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
LR = CS.Float(name="learning_rate", bounds=(0.0, 0.05))
# Value used for momentum like adaptation, as adam optimizer has no real momentum; "beta1" is changed
MOMENTUM = CS.Float(
    name="momentum", bounds=(0.0, 1.0)
)  # ! Only used, when "use_momentum" var in config true
DEFAULT_CFG_SPACE.add_hyperparameter(LR)
DEFAULT_CFG_SPACE.add_hyperparameter(MOMENTUM)


def __default_loss_function(**kwargs):
    return nn.NLLLoss(reduction="none", **kwargs)


feature_extractor = nn.Sequential(
    nn.Identity(),
    nn.Conv2d(1, 5, kernel_size=(7, 7), stride=(1, 1)),
    nn.BatchNorm2d(5, eps=1e-05, momentum=0.1),
    nn.ReLU(),
)

linear_layers = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=2420, out_features=10, bias=True),
    nn.LogSoftmax(dim=1),
)

neural_network = nn.Sequential(feature_extractor, linear_layers)  # (n_conv_layers: 5)

INFO = {
    "identifier": "LR",
    "name": "Learning Rate Adaption for Neural Networks",
    "reward": "Negative Log Differential Validation Loss",
    "state_description": [
        "Step",
        "Loss",
        "Validation Loss",
        "Crashed",
    ],
    "action_description": ["Learning Rate", "Momentum"],
}


SGD_DEFAULTS = objdict(
    {
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Dict",
        "observation_space_type": None,
        "observation_space_args": [
            {
                "step": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "loss": spaces.Box(0, np.inf, shape=(1,)),
                "validationLoss": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "crashed": spaces.Discrete(1),
            }
        ],
        "shuffle_training": True,
        "reward_range": [-(10**9), (10**9)],
        "model": neural_network,
        "optimizer_params": {
            "weight_decay": 10.978902603194243,
            "eps": 1.2346464628039852e-10,
            "betas": (0.9994264825468422, 0.9866804882743139),
        },
        "cutoff": 1e2,
        "loss_function": __default_loss_function,
        "loss_function_kwargs": {},
        "training_batch_size": 64,
        "fraction_of_dataset": 0.6,
        "train_validation_ratio": 0.8,  # If set to None, random value is used
        "dataset_name": "MNIST",  # If set to None, random data set is chosen; else specific set can be set: e.g. "MNIST"
        # "reward_function":,    # Can be set, to replace the default function
        # "state_method":,       # Can be set, to replace the default function
        "use_momentum": False,
        "seed": 0,
        "crash_penalty": 100.0,
        "multi_agent": False,
        "instance_set_path": "../instance_sets/sgd/sgd_train_100instances.csv",
        "benchmark_info": INFO,
    }
)


class SGDBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for SGD
    """

    def __init__(self, config_path=None, config=None):
        """
        Initialize SGD Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(SGDBenchmark, self).__init__(config_path, config)
        if not self.config:
            self.config = objdict(SGD_DEFAULTS.copy())

        for key in SGD_DEFAULTS:
            if key not in self.config:
                self.config[key] = SGD_DEFAULTS[key]

    def get_environment(self, use_generator=False):
        """
        Return SGDEnv env with current configuration

        Returns
        -------
        SGDEnv
            SGD environment
        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        # Read test set if path is specified
        if (
            "test_set" not in self.config.keys()
            and "test_set_path" in self.config.keys()
        ):
            self.read_instance_set(test=True)

        env = SGDEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        if use_generator:
            (
                env.model,
                env.optimizer_params,
                env.loss,
                env.batch_size,
                env.crash_penalty,
                env.n_conv_layers,
            ) = utils.random_instance(
                np.random.RandomState(self.config.get("seed")), env.datasets
            )

        return env

    def read_instance_set(self, test=False):
        """
        Read path of instances from config into list
        """

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
        with open(path, "r") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            for row in reader:
                if "_" in row["dataset"]:
                    dataset_info = row["dataset"].split("_")
                    dataset_name = dataset_info[0]
                    dataset_size = int(dataset_info[1])
                else:
                    dataset_name = row["dataset"]
                    dataset_size = None
                instance = [
                    dataset_name,
                    int(row["seed"]),
                    row["architecture"],
                    int(row["steps"]),
                    dataset_size,
                ]
                self.config[keyword][int(row["ID"])] = instance

    def get_benchmark(self, instance_set_path=None, seed=0):
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
        if instance_set_path is not None:
            self.config["instance_set_path"] = instance_set_path
        self.config.seed = seed
        self.read_instance_set()
        return SGDEnv(self.config)
