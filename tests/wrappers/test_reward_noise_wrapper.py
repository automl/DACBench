from __future__ import annotations

import unittest

import pytest
from dacbench.benchmarks import LubyBenchmark
from dacbench.wrappers import RewardNoiseWrapper


class TestRewardNoiseWrapper(unittest.TestCase):
    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = RewardNoiseWrapper(env)
        assert wrapped.noise_function is not None

        with pytest.raises(Exception):
            wrapped = RewardNoiseWrapper(env, noise_dist=None)
        with pytest.raises(Exception):
            wrapped = RewardNoiseWrapper(env, noise_dist="norm")

        wrapped = RewardNoiseWrapper(env, noise_dist="normal", dist_args=[0, 0.3])
        assert wrapped.noise_function is not None

        def dummy():
            return 0

        wrapped = RewardNoiseWrapper(env, noise_function=dummy)
        assert wrapped.noise_function is not None

    def test_step(self):
        bench = LubyBenchmark()
        bench.config.reward_range = (-10, 10)
        env = bench.get_environment()
        env.reset()
        _, raw_reward, _, _, _ = env.step(1)

        wrapped = RewardNoiseWrapper(env)
        wrapped.reset()
        _, reward, _, _, _ = wrapped.step(1)
        assert reward != raw_reward

        wrapped = RewardNoiseWrapper(env, noise_dist="normal", dist_args=[0, 0.3])
        wrapped.reset()
        env.reset()
        _, raw_reward, _, _, _ = env.step(1)
        _, reward, _, _, _ = wrapped.step(1)
        assert reward != raw_reward

        def dummy():
            return 0

        wrapped = RewardNoiseWrapper(env, noise_function=dummy)
        wrapped.reset()
        _, reward, _, _, _ = wrapped.step(1)
        assert reward in (0, -1)

    def test_getters_and_setters(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = RewardNoiseWrapper(env)

        assert wrapped.noise_function == wrapped.noise_function
        assert wrapped.env == wrapped.env

        print(wrapped.action_space)
        print(wrapped.env.action_space)
        print(wrapped.env.action_space)
        assert wrapped.action_space == wrapped.env.action_space
        assert wrapped.observation_space == wrapped.env.observation_space
        assert wrapped.reward_range == wrapped.env.reward_range
