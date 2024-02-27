from __future__ import annotations

import unittest

import numpy as np
import pytest
from dacbench import AbstractEnv
from dacbench.benchmarks.luby_benchmark import LUBY_DEFAULTS
from dacbench.envs import LubyEnv


class TestLubyEnv(unittest.TestCase):
    def make_env(self):
        config = LUBY_DEFAULTS
        config["instance_set"] = {0: [1, 1]}
        return LubyEnv(config)

    def test_setup(self):
        env = self.make_env()
        assert issubclass(type(env), AbstractEnv)
        assert env.np_random is not None
        assert env._genny is not None
        assert env._next_goal is not None
        assert env._seq is not None
        assert env._ms == LUBY_DEFAULTS["cutoff"]
        assert env._mi == LUBY_DEFAULTS["min_steps"]
        assert env._hist_len == LUBY_DEFAULTS["hist_length"]
        assert env._start_shift == 0
        assert env._sticky_shif == 0

    def test_reset(self):
        env = self.make_env()
        state, info = env.reset()
        assert issubclass(type(info), dict)
        assert env._start_shift, 1
        assert env._sticky_shif, 1
        assert np.array_equal(-1 * np.ones(LUBY_DEFAULTS["hist_length"] + 1), state)

    def test_step(self):
        env = self.make_env()
        env.reset()
        state, reward, terminated, truncated, meta = env.step(1)
        assert reward >= env.reward_range[0]
        assert reward <= env.reward_range[1]
        assert state[-1] == 0
        assert state[0] == 1
        assert np.array_equal(state[1:-1], -1 * np.ones(4))
        assert len(state) == env._hist_len + 1
        assert not terminated
        assert not truncated
        assert len(meta.keys()) == 0

        config = LUBY_DEFAULTS
        config["instance_set"] = {1: [-4, -4]}
        env = LubyEnv(config)
        env.reset()
        state, reward, terminated, truncated, meta = env.step(1)
        assert reward >= env.reward_range[0]
        assert reward <= env.reward_range[1]
        assert state[-1] == 0
        assert state[0] == 1
        assert np.array_equal(state[1:-1], -1 * np.ones(4))
        assert len(state) == env._hist_len + 1
        assert not terminated
        assert not truncated
        assert len(meta.keys()) == 0

    def test_close(self):
        env = self.make_env()
        assert env.close()

    def test_render(self):
        env = self.make_env()
        env.render("human")
        with pytest.raises(NotImplementedError):
            env.render("random")
