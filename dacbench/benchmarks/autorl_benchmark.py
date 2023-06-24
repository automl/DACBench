import csv
import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import AutoRLEnv


DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
LR = CSH.UniformFloatHyperparameter(
    name="lr", lower=0, upper=0.9
)
DEFAULT_CFG_SPACE.add_hyperparameter(LR)

DEFAULT_RL_CONFIG = {
            "lr": 2.5e-4,
            "num_envs": 4,
            "num_steps": 128,
            "total_timesteps": 100,
            "update_epochs": 4,
            "num_minibatches": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "activation": "tanh",
            "env_name": "CartPole-v1", 
            "num_eval_episodes": 10,
            "env_framework": "gymnax"}

INFO = {
    "identifier": "AutoRL",
    "name": "Hyperparameter Control for Reinforcement Learning",
    "reward": "Evaluation Reward",
    "state_description": [
    ],
}

AUTORL_DEFAULTS = objdict(
    {
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [np.array([-1]), np.array([1])],
        "reward_range": (-np.inf, np.inf),
        "cutoff": 1000,
        "seed": 0,
        "benchmark_info": INFO,
        "checkpoint": True,
        "checkpoint_dir": "autorl_checkpoints",
        "instance_set": {0: DEFAULT_RL_CONFIG},
        "track_trajectory": False,
        "grad_obs": False,
        "algorithm": "ppo"
    }
)


class AutoRLBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for Sigmoid
    """

    def __init__(self, config_path=None, config=None):
        """
        Initialize Luby Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super().__init__(config_path, config)
        if not self.config:
            self.config = objdict(AUTORL_DEFAULTS.copy())

        for key in AUTORL_DEFAULTS:
            if key not in self.config:
                self.config[key] = AUTORL_DEFAULTS[key]

    def get_environment(self):
        """
        Return Luby env with current configuration

        Returns
        -------
        LubyEnv
            Luby environment
        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        # Read test set if path is specified
        if (
            "test_set" not in self.config.keys()
            and "test_set_path" in self.config.keys()
        ):
            self.read_instance_set(test=True)

        env = AutoRLEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

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
        with open(path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self.config[keyword][int(row["ID"])] = [
                    float(shift) for shift in row["start"].split(",")
                ] + [float(slope) for slope in row["sticky"].split(",")]
