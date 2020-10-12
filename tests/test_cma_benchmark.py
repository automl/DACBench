import unittest
from daclib.benchmarks import CMAESBenchmark
from daclib.envs import CMAESEnv


class TestCMABenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = CMAESBenchmark()
        env = bench.get_benchmark_env()
        self.assertTrue(issubclass(type(env), CMAESEnv))

    def test_setup(self):
        bench = CMAESBenchmark()
        self.assertTrue(bench.config is not None)
