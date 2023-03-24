import csv
import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from gymnasium import spaces
from torch.nn import NLLLoss

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import SGDEnv
from dacbench.envs.sgd import Reward

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
LR = CSH.UniformIntegerHyperparameter(name="learning_rate", lower=0, upper=10)
DEFAULT_CFG_SPACE.add_hyperparameter(LR)


def __default_loss_function(**kwargs):
    return NLLLoss(reduction="none", **kwargs)


INFO = {
    "identifier": "LR",
    "name": "Learning Rate Adaption for Neural Networks",
    "reward": "Negative Log Differential Validation Loss",
    "state_description": [
        "Predictive Change Variance (Discounted Average)",
        "Predictive Change Variance (Uncertainty)",
        "Loss Variance (Discounted Average)",
        "Loss Variance (Uncertainty)",
        "Current Learning Rate",
        "Training Loss",
        "Validation Loss",
        "Step",
        "Alignment",
        "Crashed",
    ],
}


SGD_DEFAULTS = objdict(
    {
        "config_space": DEFAULT_CFG_SPACE,
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
                "lossVarDiscountedAverage": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,)
                ),
                "lossVarUncertainty": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "currentLR": spaces.Box(low=0, high=1, shape=(1,)),
                "trainingLoss": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "validationLoss": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "step": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "alignment": spaces.Box(low=0, high=1, shape=(1,)),
                "crashed": spaces.Discrete(2),
            }
        ],
        "reward_type": Reward.LogDiffTraining,
        "cutoff": 1e3,
        "lr": 1e-3,
        "discount_factor": 0.9,
        "optimizer": "rmsprop",
        "loss_function": __default_loss_function,
        "loss_function_kwargs": {},
        "val_loss_function": __default_loss_function,
        "val_loss_function_kwargs": {},
        "training_batch_size": 64,
        "validation_batch_size": 64,
        "train_validation_ratio": 0.8,
        "dataloader_shuffle": True,
        "no_cuda": False,
        "beta1": 0.9,
        "beta2": 0.9,
        "epsilon": 1.0e-06,
        "clip_grad": (-1.0, 1.0),
        "seed": 0,
        "cd_paper_reconstruction": False,
        "cd_bias_correction": True,
        "terminate_on_crash": False,
        "crash_penalty": 0.0,
        "instance_set_path": "../instance_sets/sgd/sgd_train_100instances.csv",
        "benchmark_info": INFO,
        "features": [
            "predictiveChangeVarDiscountedAverage",
            "predictiveChangeVarUncertainty",
            "lossVarDiscountedAverage",
            "lossVarUncertainty",
            "currentLR",
            "trainingLoss",
            "validationLoss",
            "step",
            "alignment",
            "crashed",
        ],
    }
)

# Set reward range based on the chosen reward type
SGD_DEFAULTS.reward_range = SGD_DEFAULTS["reward_type"].func.frange


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

        # Read test set if path is specified
        if (
            "test_set" not in self.config.keys()
            and "test_set_path" in self.config.keys()
        ):
            self.read_instance_set(test=True)

        env = SGDEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

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
