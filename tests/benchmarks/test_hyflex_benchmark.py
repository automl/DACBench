import unittest
import json
import os

from dacbench.benchmarks import HyFlexBenchmark
from dacbench.envs import HyFlexEnv


class TestHyFlexBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = HyFlexBenchmark()
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), HyFlexEnv))    

    def test_save_conf(self):
        bench = HyFlexBenchmark()
        bench.save_config("test_conf.json")
        with open("test_conf.json", "r") as fp:
            recovered = json.load(fp)
        for k in bench.config.keys():
            self.assertTrue(k in recovered.keys())
        os.remove("test_conf.json")

    def test_read_instances(self):
        bench = HyFlexBenchmark()
        bench.read_instance_set()        
        first_inst = bench.config.instance_set[0]

        bench2 = HyFlexBenchmark()
        env = bench2.get_environment()
        self.assertTrue(env.instance_set[0] == first_inst)
        self.assertTrue(len(env.instance_set) == len(bench.config.instance_set)) 
