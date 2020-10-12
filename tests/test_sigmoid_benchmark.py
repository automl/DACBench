import unittest
from daclib.benchmarks import SigmoidBenchmark
from daclib.envs import SigmoidEnv


class TestSigmoidBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = SigmoidBenchmark()
        env = bench.get_benchmark_env()
        self.assertTrue(issubclass(type(env), SigmoidEnv))

    def test_setup(self):
        bench = SigmoidBenchmark()
        self.assertTrue(bench.config is not None)
