import unittest
import os
import json
from dacbench.benchmarks import FastDownwardBenchmark
from dacbench.envs import FastDownwardEnv


class TestFDBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = FastDownwardBenchmark()
        env = bench.get_benchmark_env()
        self.assertTrue(issubclass(type(env), FastDownwardEnv))

        bench.config.instance_set_path = "../instance_sets/fast_downward/childsnack"
        bench.read_instance_set()
        env = bench.get_benchmark_env()
        self.assertTrue(issubclass(type(env), FastDownwardEnv))

    def test_setup(self):
        bench = FastDownwardBenchmark()
        self.assertTrue(bench.config is not None)

        config = {"dummy": 0}
        with open("test_conf.json", "w+") as fp:
            json.dump(config, fp)
        bench = FastDownwardBenchmark("test_conf.json")
        self.assertTrue(bench.config.dummy == 0)
        os.remove("test_conf.json")

    def test_read_instances(self):
        bench = FastDownwardBenchmark()
        bench.read_instance_set()
        self.assertTrue(len(bench.config.instance_set) == 30)
        self.assertTrue(type(bench.config.instance_set[0]) == str)
        self.assertTrue(os.path.isfile(bench.config.instance_set[0]))
        path = bench.config.instance_set[0]
        bench2 = FastDownwardBenchmark()
        env = bench2.get_benchmark_env()
        self.assertTrue(type(env.instance_set[0]) == str)
        self.assertTrue(len(env.instance_set) == 30)
        self.assertTrue(path == env.instance_set[0])

    def test_benchmark_env(self):
        bench = FastDownwardBenchmark()
        env = bench.get_benchmark()
        self.assertTrue(issubclass(type(env), FastDownwardEnv))
