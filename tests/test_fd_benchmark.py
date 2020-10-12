import unittest
import os
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

    def test_read_instances(self):
        bench = FastDownwardBenchmark()
        bench.read_instance_set()
        self.assertTrue(len(bench.config.instance_set)==30)
        self.assertTrue(type(bench.config.instance_set[0])==str)
        self.assertTrue(os.path.isfile(bench.config.instance_set[0]))
        path = bench.config.instance_set[0]
        bench2 = FastDownwardBenchmark()
        env = bench2.get_benchmark_env()
        self.assertTrue(type(env.instance_set[0])==str)
        self.assertTrue(len(env.instance_set)==30)
        self.assertTrue(path == env.instance_set[0])
