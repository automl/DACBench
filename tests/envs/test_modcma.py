import unittest

import numpy as np
from gymnasium import spaces

from dacbench import AbstractEnv
from dacbench.abstract_benchmark import objdict
from dacbench.envs import ModCMAEnv


class TestModCMAEnv(unittest.TestCase):
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
        env = ModCMAEnv(config)
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
        state, reward, terminated, truncated, meta = env.step(np.ones(11, dtype=int))
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
