import unittest
import json
import os

from dacbench.benchmarks import ModeaBenchmark
from dacbench.envs import ModeaEnv


class TestCMABenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = ModeaBenchmark()
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), ModeaEnv))

    def test_setup(self):
        bench = ModeaBenchmark()
        self.assertTrue(bench.config is not None)

        config = {"dummy": 0}
        with open("test_conf.json", "w+") as fp:
            json.dump(config, fp)
        bench = ModeaBenchmark("test_conf.json")
        self.assertTrue(bench.config.dummy == 0)
        os.remove("test_conf.json")

    def test_read_instances(self):
        bench = ModeaBenchmark()
        bench.read_instance_set()
        self.assertTrue(len(bench.config.instance_set) == 100)
        inst = bench.config.instance_set[0]
        bench2 = ModeaBenchmark()
        env = bench2.get_environment()
        self.assertTrue(len(env.instance_set) == 100)
        self.assertTrue(inst == env.instance_set[0])

    def test_benchmark_env(self):
        bench = ModeaBenchmark()
        env = bench.get_benchmark()
        self.assertTrue(issubclass(type(env), ModeaEnv))
