import unittest
import numpy as np
from dacbench import AbstractEnv
from dacbench.envs import ModeaEnv
from dacbench.abstract_benchmark import objdict
from gym import spaces


class TestModeaEnv(unittest.TestCase):
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
        config.reward_range = (-(10 ** 12), 0)
        env = ModeaEnv(config)
        return env

    def test_setup(self):
        env = self.make_env()
        self.assertTrue(issubclass(type(env), AbstractEnv))

    def test_reset(self):
        env = self.make_env()
        env.reset()

    def test_step(self):
        env = self.make_env()
        env.reset()
        state, reward, done, meta = env.step(np.ones(14))
        self.assertTrue(reward >= env.reward_range[0])
        self.assertTrue(reward <= env.reward_range[1])
        self.assertFalse(done)
        self.assertTrue(len(meta.keys()) == 0)
        self.assertTrue(len(state) == 5)
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())

    def test_close(self):
        env = self.make_env()
        self.assertTrue(env.close())
