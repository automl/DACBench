import unittest
from unittest import mock

import numpy as np

from dacbench import AbstractEnv
from dacbench.benchmarks.sigmoid_benchmark import SIGMOID_DEFAULTS
from dacbench.envs import SigmoidEnv


class TestSigmoidEnv(unittest.TestCase):
    def make_env(self):
        config = SIGMOID_DEFAULTS
        config["instance_set"] = {20: [0, 1, 2, 3]}
        env = SigmoidEnv(config)
        return env

    def test_setup(self):
        env = self.make_env()
        self.assertTrue(issubclass(type(env), AbstractEnv))
        self.assertFalse(env.np_random is None)
        self.assertTrue(
            np.array_equal(
                env.shifts, 5 * np.ones(len(SIGMOID_DEFAULTS["action_values"]))
            )
        )
        self.assertTrue(
            np.array_equal(
                env.slopes, -1 * np.ones(len(SIGMOID_DEFAULTS["action_values"]))
            )
        )
        self.assertTrue(env.n_actions == len(SIGMOID_DEFAULTS["action_values"]))
        self.assertTrue(env.slope_multiplier == SIGMOID_DEFAULTS["slope_multiplier"])
        self.assertTrue(
            (env.action_space.nvec + 1 == SIGMOID_DEFAULTS["action_values"]).all()
        )

    def test_reset(self):
        env = self.make_env()
        state, info = env.reset()
        self.assertTrue(issubclass(type(info), dict))
        self.assertTrue(np.array_equal(env.shifts, [0, 1]))
        self.assertTrue(np.array_equal(env.slopes, [2, 3]))
        self.assertTrue(state[0] == SIGMOID_DEFAULTS["cutoff"])
        self.assertTrue(np.array_equal([state[1], state[3]], env.shifts))
        self.assertTrue(np.array_equal([state[2], state[4]], env.slopes))
        self.assertTrue(np.array_equal(state[5:], -1 * np.ones(2)))

    def test_step(self):
        env = self.make_env()
        env.reset()
        state, reward, terminated, truncated, meta = env.step([1, 1])
        self.assertTrue(reward >= env.reward_range[0])
        self.assertTrue(reward <= env.reward_range[1])
        self.assertTrue(state[0] == 9)
        self.assertTrue(np.array_equal([state[1], state[3]], env.shifts))
        self.assertTrue(np.array_equal([state[2], state[4]], env.slopes))
        self.assertTrue(len(state) == 7)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(len(meta.keys()) == 0)

    def test_close(self):
        env = self.make_env()
        self.assertTrue(env.close())

    @mock.patch("dacbench.envs.sigmoid.plt")
    def test_render(self, mock_plt):
        env = self.make_env()
        env.render("random")
        self.assertFalse(mock_plt.show.called)
        env.render("human")
        self.assertTrue(mock_plt.show.called)
