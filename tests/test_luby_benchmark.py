import unittest
from daclib.benchmarks import LubyBenchmark
from daclib.envs import LubyEnv


class TestLubyBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        self.assertTrue(issubclass(type(env), LubyEnv))

    def test_setup(self):
        bench = LubyBenchmark()
        self.assertTrue(bench.config is not None)
