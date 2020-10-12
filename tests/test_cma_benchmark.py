import unittest
import numpy as np
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

    def test_read_instances(self):
        bench = CMAESBenchmark()
        bench.read_instance_set()
        self.assertTrue(len(bench.config.instance_set)==100)
        self.assertTrue(len(bench.config.instance_set[0])==4)
        self.assertTrue(bench.config.instance_set[0][2]==0.6445072293504781)
        inst = bench.config.instance_set[0]
        bench2 = CMAESBenchmark()
        env = bench2.get_benchmark_env()
        self.assertTrue(len(env.instance_set[0])==4)
        self.assertTrue(len(env.instance_set)==100)
        self.assertTrue(inst == env.instance_set[0])
