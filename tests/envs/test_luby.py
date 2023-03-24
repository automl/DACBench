import unittest

import numpy as np

from dacbench import AbstractEnv
from dacbench.benchmarks.luby_benchmark import LUBY_DEFAULTS
from dacbench.envs import LubyEnv


class TestLubyEnv(unittest.TestCase):
    def make_env(self):
        config = LUBY_DEFAULTS
        config["instance_set"] = {0: [1, 1]}
        env = LubyEnv(config)
        return env

    def test_setup(self):
        env = self.make_env()
        self.assertTrue(issubclass(type(env), AbstractEnv))
        self.assertFalse(env.np_random is None)
        self.assertFalse(env._genny is None)
        self.assertFalse(env._next_goal is None)
        self.assertFalse(env._seq is None)
        self.assertTrue(env._ms == LUBY_DEFAULTS["cutoff"])
        self.assertTrue(env._mi == LUBY_DEFAULTS["min_steps"])
        self.assertTrue(env._hist_len == LUBY_DEFAULTS["hist_length"])
        self.assertTrue(env._start_shift == 0)
        self.assertTrue(env._sticky_shif == 0)

    def test_reset(self):
        env = self.make_env()
        state, info = env.reset()
        self.assertTrue(issubclass(type(info), dict))
        self.assertTrue(env._start_shift, 1)
        self.assertTrue(env._sticky_shif, 1)
        self.assertTrue(
            np.array_equal(-1 * np.ones(LUBY_DEFAULTS["hist_length"] + 1), state)
        )

    def test_step(self):
        env = self.make_env()
        env.reset()
        state, reward, terminated, truncated, meta = env.step(1)
        self.assertTrue(reward >= env.reward_range[0])
        self.assertTrue(reward <= env.reward_range[1])
        self.assertTrue(state[-1] == 0)
        self.assertTrue(state[0] == 1)
        self.assertTrue(np.array_equal(state[1:-1], -1 * np.ones(4)))
        self.assertTrue(len(state) == env._hist_len + 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(len(meta.keys()) == 0)

        config = LUBY_DEFAULTS
        config["instance_set"] = {1: [-4, -4]}
        env = LubyEnv(config)
        env.reset()
        state, reward, terminated, truncated, meta = env.step(1)
        self.assertTrue(reward >= env.reward_range[0])
        self.assertTrue(reward <= env.reward_range[1])
        self.assertTrue(state[-1] == 0)
        self.assertTrue(state[0] == 1)
        self.assertTrue(np.array_equal(state[1:-1], -1 * np.ones(4)))
        self.assertTrue(len(state) == env._hist_len + 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(len(meta.keys()) == 0)

    def test_close(self):
        env = self.make_env()
        self.assertTrue(env.close())

    def test_render(self):
        env = self.make_env()
        env.render("human")
        with self.assertRaises(NotImplementedError):
            env.render("random")
