import json
import os
import unittest

from dacbench.benchmarks import CMAESBenchmark
from dacbench.envs import CMAESEnv


class TestCMABenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = CMAESBenchmark()
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), CMAESEnv))

    def test_setup(self):
        bench = CMAESBenchmark()
        self.assertTrue(bench.config is not None)

        config = {"dummy": 0}
        with open("test_conf.json", "w+") as fp:
            json.dump(config, fp)
        bench = CMAESBenchmark("test_conf.json")
        self.assertTrue(bench.config.dummy == 0)
        os.remove("test_conf.json")

    def test_save_conf(self):
        bench = CMAESBenchmark()
        del bench.config["config_space"]
        bench.save_config("test_conf.json")
        with open("test_conf.json", "r") as fp:
            recovered = json.load(fp)
        for k in bench.config.keys():
            self.assertTrue(k in recovered.keys())
        os.remove("test_conf.json")

    def test_from_to_json(self):
        bench = CMAESBenchmark()
        restored_bench = CMAESBenchmark.from_json(bench.to_json())
        self.assertEqual(bench, restored_bench)

    def test_read_instances(self):
        bench = CMAESBenchmark()
        bench.read_instance_set()
        self.assertTrue(len(bench.config.instance_set.keys()) == 100)
        self.assertTrue(len(bench.config.instance_set[0]) == 4)
        self.assertTrue(bench.config.instance_set[0][2] == 0.6445072293504781)
        inst = bench.config.instance_set[0]
        bench2 = CMAESBenchmark()
        env = bench2.get_environment()
        self.assertTrue(len(env.instance_set[0]) == 4)
        self.assertTrue(len(env.instance_set.keys()) == 100)
        self.assertTrue(inst == env.instance_set[0])

    def test_benchmark_env(self):
        bench = CMAESBenchmark()
        env = bench.get_benchmark()
        self.assertTrue(issubclass(type(env), CMAESEnv))
