import pytest
import unittest

import numpy as np
from gym import spaces
from daclib.benchmarks import LubyBenchmark
from daclib.wrappers import RewardNoiseWrapper


class TestRewardNoiseWrapper(unittest.TestCase):
    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = RewardNoiseWrapper(env)
        self.assertFalse(wrapped.noise_function is None)

        with pytest.raises(Exception):
            wrapped = RewardNoiseWrapper(env, noise_dist=None)
        with pytest.raises(Exception):
            wrapped = RewardNoiseWrapper(env, noise_dist="norm")

        wrapped = RewardNoiseWrapper(env, noise_dist="normal", dist_args=[0, 0.3])
        self.assertFalse(wrapped.noise_function is None)

        def dummy():
            return 0

        wrapped = RewardNoiseWrapper(env, noise_function=dummy)
        self.assertFalse(wrapped.noise_function is None)

    def test_step(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        env.reset()
        _, raw_reward, _, _ = env.step(1)

        wrapped = RewardNoiseWrapper(env)
        wrapped.reset()
        _, reward, _, _ = wrapped.step(1)
        self.assertTrue(reward != raw_reward)

        wrapped = RewardNoiseWrapper(env, noise_dist="normal", dist_args=[0, 0.3])
        wrapped.reset()
        _, reward, _, _ = wrapped.step(1)
        self.assertTrue(reward != raw_reward)

        def dummy():
            return 0

        wrapped = RewardNoiseWrapper(env, noise_function=dummy)
        wrapped.reset()
        _, reward, _, _ = wrapped.step(1)
        self.assertTrue(reward == raw_reward)
