import os

import numpy as np
import pandas as pd
from gym import spaces

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
import importlib
import dacbench.envs.toysgd
importlib.reload(dacbench.envs.toysgd)

INFO = {
    "identifier": "toy_sgd",
    "name": "Learning Rate and Momentum Adaption for SGD on Toy Functions",
    "reward": "Negative Log Regret",
    "state_description": [
        "Remaining Budget",
        "Gradient",
        "Current Learning Rate",
        "Current Momentum"
    ],
}

DEFAULTS = objdict(
    {
        "action_space_class": "Box",
        "action_space_args": [-np.inf * np.ones((2,)), np.ones((2,))],
        "observation_space_class": "Dict",
        "observation_space_type": None,
        "observation_space_args": [
            {
                "remaining_budget": spaces.Box(low=0, high=np.inf, shape=(1,)),
                "gradient": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                "learning_rate": spaces.Box(low=0, high=1, shape=(1,)),
                "momentum": spaces.Box(low=0, high=1, shape=(1,)),
            }
        ],
        "reward_range": (-np.inf, np.inf),
        "cutoff": 10,
        "seed": 0,
        "instance_set_path": "../instance_sets/toysgd/toysgd_default.csv",
        "benchmark_info": INFO,
    }
)


class ToySGDBenchmark(AbstractBenchmark):
    def __init__(self, config_path=None):
        """
        Initialize SGD Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(ToySGDBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = objdict(DEFAULTS.copy())

        for key in DEFAULTS:
            if key not in self.config:
                self.config[key] = DEFAULTS[key]

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

        env = dacbench.envs.toysgd.ToySGDEnv(self.config)
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
            # reader = csv.DictReader(fh, delimiter=";")
            df = pd.read_csv(fh, sep=";")
            for index, instance in df.iterrows():
                self.config["instance_set"][int(instance["ID"])] = instance

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
        self.config = objdict(DEFAULTS.copy())
        self.config.seed = seed
        self.read_instance_set()
        return dacbench.envs.toysgd.ToySGDEnv(self.config)


if __name__ == '__main__':
    benchmark = ToySGDBenchmark()
    env = benchmark.get_environment()
    log_learning_rate = -10
    log_momentum = 0
    action = (log_learning_rate, log_momentum)
    action = log_learning_rate
    for i in range(5):
        state = env.reset()
        done = False
        while not done:
            state, reward, done, info = env.step(action)
            print(state, reward)
        env.render()
    env.close()
