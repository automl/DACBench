import unittest

from dacbench.benchmarks import LubyBenchmark
from dacbench.wrappers import RewardNoiseWrapper


class TestRewardNoiseWrapper(unittest.TestCase):
    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = RewardNoiseWrapper(env)
        self.assertFalse(wrapped.noise_function is None)

        with self.assertRaises(Exception):
            wrapped = RewardNoiseWrapper(env, noise_dist=None)
        with self.assertRaises(Exception):
            wrapped = RewardNoiseWrapper(env, noise_dist="norm")

        wrapped = RewardNoiseWrapper(env, noise_dist="normal", dist_args=[0, 0.3])
        self.assertFalse(wrapped.noise_function is None)

        def dummy():
            return 0

        wrapped = RewardNoiseWrapper(env, noise_function=dummy)
        self.assertFalse(wrapped.noise_function is None)

    def test_step(self):
        bench = LubyBenchmark()
        bench.config.reward_range = (-10, 10)
        env = bench.get_environment()
        env.reset()
        _, raw_reward, _, _, _ = env.step(1)

        wrapped = RewardNoiseWrapper(env)
        wrapped.reset()
        _, reward, _, _, _ = wrapped.step(1)
        self.assertTrue(reward != raw_reward)

        wrapped = RewardNoiseWrapper(env, noise_dist="normal", dist_args=[0, 0.3])
        wrapped.reset()
        env.reset()
        _, raw_reward, _, _, _ = env.step(1)
        _, reward, _, _, _ = wrapped.step(1)
        self.assertTrue(reward != raw_reward)

        def dummy():
            return 0

        wrapped = RewardNoiseWrapper(env, noise_function=dummy)
        wrapped.reset()
        _, reward, _, _, _ = wrapped.step(1)
        self.assertTrue(reward == 0 or reward == -1)

    def test_getters_and_setters(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = RewardNoiseWrapper(env)

        self.assertTrue(wrapped.noise_function == getattr(wrapped, "noise_function"))
        self.assertTrue(wrapped.env == getattr(wrapped, "env"))

        print(wrapped.action_space)
        print(wrapped.env.action_space)
        print(getattr(wrapped.env, "action_space"))
        self.assertTrue(wrapped.action_space == getattr(wrapped.env, "action_space"))
        self.assertTrue(
            wrapped.observation_space == getattr(wrapped.env, "observation_space")
        )
        self.assertTrue(wrapped.reward_range == getattr(wrapped.env, "reward_range"))
