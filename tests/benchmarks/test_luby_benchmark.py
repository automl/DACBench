import unittest
import numpy as np
from daclib.benchmarks import LubyBenchmark
from daclib.envs import LubyEnv
from daclib.wrappers import RewardNoiseWrapper


class TestLubyBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        self.assertTrue(issubclass(type(env), LubyEnv))

    def test_setup(self):
        bench = LubyBenchmark()
        self.assertTrue(bench.config is not None)

    def test_read_instances(self):
        bench = LubyBenchmark()
        bench.read_instance_set()
        self.assertTrue(len(bench.config.instance_set) == 100)
        self.assertTrue(len(bench.config.instance_set[0]) == 2)
        self.assertTrue(bench.config.instance_set[0] == [34, -0.07])
        bench2 = LubyBenchmark()
        env = bench2.get_benchmark_env()
        self.assertTrue(len(env.instance_set[0]) == 2)
        self.assertTrue(env.instance_set[0] == [34, -0.07])
        self.assertTrue(len(env.instance_set) == 100)

    def test_benchmark_env(self):
        bench = LubyBenchmark()
        env = bench.get_complete_benchmark()
        self.assertTrue(issubclass(type(env), RewardNoiseWrapper))

    def test_cutoff_setting(self):
        bench = LubyBenchmark()
        bench.set_cutoff(100)
        self.assertTrue(bench.config.cutoff == 100)
        self.assertTrue(bench.config.action_space_args == [int(np.log2(100))])

    def test_history_len_setting(self):
        bench = LubyBenchmark()
        bench.set_history_length(20)
        self.assertTrue(len(bench.config.observation_space_args[0]) == 21)
