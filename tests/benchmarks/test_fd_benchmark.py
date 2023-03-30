import json
import os
import unittest

from dacbench.benchmarks import FastDownwardBenchmark
from dacbench.envs import FastDownwardEnv


class TestFDBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = FastDownwardBenchmark()
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), FastDownwardEnv))

        bench.config.instance_set_path = "../instance_sets/fast_downward/childsnack"
        bench.read_instance_set()
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), FastDownwardEnv))

    # TODO: This test breaks remote testing, possibly due to too many open ports.
    # Should be investigated
    # def test_scenarios(self):
    #     scenarios = [
    #         "fd_barman.json",
    #         # "fd_blocksworld.json",
    #         # "fd_visitall.json",
    #         # "fd_childsnack.json",
    #         # "fd_sokoban.json",
    #         # "fd_rovers.json",
    #     ]
    #     for s in scenarios:
    #         path = os.path.join("dacbench/additional_configs/fast_downward/", s)
    #         bench = FastDownwardBenchmark(path)
    #         self.assertTrue(bench.config is not None)
    #         env = bench.get_environment()
    #         state, info = env.reset()
    #         self.assertTrue(state is not None)
    #         self.assertTrue(info is not None)
    #         state, _, _, _, _ = env.step(0)
    #         self.assertTrue(state is not None)

    def test_save_conf(self):
        bench = FastDownwardBenchmark()
        del bench.config["config_space"]
        bench.save_config("test_conf.json")
        with open("test_conf.json", "r") as fp:
            recovered = json.load(fp)
        for k in bench.config.keys():
            self.assertTrue(k in recovered.keys())
        os.remove("test_conf.json")

    def test_read_instances(self):
        bench = FastDownwardBenchmark()
        bench.read_instance_set()
        self.assertTrue(len(bench.config.instance_set.keys()) == 30)
        self.assertTrue(type(bench.config.instance_set[0]) == str)
        self.assertTrue(os.path.isfile(bench.config.instance_set[0]))
        path = bench.config.instance_set[0]
        bench2 = FastDownwardBenchmark()
        env = bench2.get_environment()
        self.assertTrue(type(env.instance_set[0]) == str)
        self.assertTrue(len(env.instance_set.keys()) == 30)
        self.assertTrue(path == env.instance_set[0])

    def test_benchmark_env(self):
        bench = FastDownwardBenchmark()
        env = bench.get_benchmark()
        self.assertTrue(issubclass(type(env), FastDownwardEnv))

    def test_from_to_json(self):
        bench = FastDownwardBenchmark()
        restored_bench = FastDownwardBenchmark.from_json(bench.to_json())
        self.assertEqual(bench, restored_bench)
