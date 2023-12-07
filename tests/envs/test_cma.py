import unittest
from collections import OrderedDict

import numpy as np
from gymnasium import spaces

from dacbench import AbstractEnv
from dacbench.abstract_benchmark import objdict
from dacbench.envs import CMAESEnv


class TestCMAEnv(unittest.TestCase):
    def make_env(self):
        config = objdict({})
        config.budget = 20
        config.datapath = "."
        config.threshold = 1e-8
        config.instance_set = {2: [10, 12, 0, np.ones(11)]}
        config.cutoff = 10
        config.benchmark_info = None
        config.action_space = spaces.MultiDiscrete([2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3])
        config.observation_space = spaces.Box(
            low=-np.inf * np.ones(5), high=np.inf * np.ones(5)
        )
        config.reward_range = (-(10**12), 0)
        env = CMAESEnv(config)
        return env

    def test_setup(self):
        env = self.make_env()
        self.assertTrue(issubclass(type(env), AbstractEnv))

    def test_reset(self):
        env = self.make_env()
        state, info = env.reset()
        self.assertTrue(issubclass(type(info), dict))
        self.assertTrue(state is not None)

    def test_step(self):
        env = self.make_env()
        env.reset()
        param_keys = (
            "active",
            "elitist",
            "orthogonal",
            "sequential",
            "threshold_convergence",
            "step_size_adaptation",
            "mirrored",
            "base_sampler",
            "weights_option",
            "local_restart",
            "bound_correction",
        )
        rand_dict = {key: 1 for key in param_keys}
        state, reward, terminated, truncated, meta = env.step(
            OrderedDict(rand_dict)
        )  # env.step(np.ones(12, dtype=int))
        self.assertTrue(reward >= env.reward_range[0])
        self.assertTrue(reward <= env.reward_range[1])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(len(meta.keys()) == 0)
        self.assertTrue(len(state) == 5)
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())

    def test_close(self):
        env = self.make_env()
        self.assertTrue(env.close())
