from __future__ import annotations

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
        return SigmoidEnv(config)

    def test_setup(self):
        env = self.make_env()
        assert issubclass(type(env), AbstractEnv)
        assert env.np_random is not None
        assert np.array_equal(
            env.shifts, 5 * np.ones(len(SIGMOID_DEFAULTS["action_values"]))
        )
        assert np.array_equal(
            env.slopes, -1 * np.ones(len(SIGMOID_DEFAULTS["action_values"]))
        )
        assert env.n_actions == len(SIGMOID_DEFAULTS["action_values"])
        assert env.slope_multiplier == SIGMOID_DEFAULTS["slope_multiplier"]
        assert (env.action_space.nvec + 1 == SIGMOID_DEFAULTS["action_values"]).all()

    def test_reset(self):
        env = self.make_env()
        state, info = env.reset()
        assert issubclass(type(info), dict)
        assert np.array_equal(env.shifts, [0, 1])
        assert np.array_equal(env.slopes, [2, 3])
        assert state[0] == SIGMOID_DEFAULTS["cutoff"]
        assert np.array_equal([state[1], state[3]], env.shifts)
        assert np.array_equal([state[2], state[4]], env.slopes)
        assert np.array_equal(state[5:], -1 * np.ones(2))

    def test_step(self):
        env = self.make_env()
        env.reset()
        state, reward, terminated, truncated, meta = env.step([1, 1])
        assert reward >= env.reward_range[0]
        assert reward <= env.reward_range[1]
        assert state[0] == 9
        assert np.array_equal([state[1], state[3]], env.shifts)
        assert np.array_equal([state[2], state[4]], env.slopes)
        assert len(state) == 7
        assert not terminated
        assert not truncated
        assert len(meta.keys()) == 0

    def test_close(self):
        env = self.make_env()
        assert env.close()

    @mock.patch("dacbench.envs.sigmoid.plt")
    def test_render(self, mock_plt):
        env = self.make_env()
        env.render("random")
        assert not mock_plt.show.called
        env.render("human")
        assert mock_plt.show.called
