import unittest
from daclib.benchmarks import FastDownwardBenchmark
from daclib.envs import FastDownwardEnv


class TestFDBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = FastDownwardBenchmark()
        env = bench.get_benchmark_env()
        self.assertTrue(issubclass(type(env), FastDownwardEnv))

    def test_setup(self):
        bench = FastDownwardBenchmark()
        self.assertTrue(bench.config is not None)
