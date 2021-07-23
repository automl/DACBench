from posixpath import abspath
from typing import Dict
import unittest
import json
import os

import numpy as np
from dacbench.abstract_benchmark import objdict
from dacbench.benchmarks import GeometricBenchmark
from dacbench.envs import GeometricEnv


DEFAULTS_DYNAMIC = objdict(
    {
        "action_space_class": "Discrete",
        "action_space_args": [],
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [],
        "reward_range": (0, 1),
        "cutoff": 10,
        "action_values": [],
        "action_value_default": 4,
        "action_values_variable": True,  # if True action value mapping will be used
        "action_value_mapping": {  # defines number of action values for differnet functions
            "sigmoid": 3,
            "linear": 3,
            "polynomial2D": 5,
            "polynomial3D": 7,
            "polynomial7D": 11,
            "exponential": 4,
            "logarithmic": 4,
        },
        "action_interval_mapping": {},  # maps actions to equally sized intervalls in [-1, 1]
        "seed": 0,
        "max_function_value": 10000,  # clip function value if it is higher than this number
        "instance_set_path": "../instance_sets/geometric/geometric_unit_test.csv",
    }
)


DEFAULTS_STATIC = objdict(
    {
        "action_space_class": "Discrete",
        "action_space_args": [],
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [],
        "reward_range": (0, 1),
        "cutoff": 10,
        "action_values": [],
        "action_value_default": 4,
        "action_values_variable": False,  # if True action value mapping will be used
        "action_interval_mapping": {},  # maps actions to equally sized intervalls in [-1, 1]
        "max_function_value": 10000,  # clip function value if it is higher than this number
        "instance_set_path": "../instance_sets/geometric/geometric_unit_test.csv",
    }
)


class TestSigmoidBenchmark(unittest.TestCase):
    def load_bench(self, config: Dict):
        with open("test_config.json", "w+") as fp:
            json.dump(config, fp)

        absolute_path = os.path.abspath("test_conf.json")
        bench = GeometricBenchmark(absolute_path)
        os.remove("test_config.json")
        return bench

    def test_get_env(self):
        bench = GeometricBenchmark()
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), GeometricEnv))

    def test_setup(self):
        bench = GeometricBenchmark()
        self.assertTrue(bench.config is not None)

        config = {"dummy": 0}
        with open("test_conf.json", "w+") as fp:
            json.dump(config, fp)
        bench = GeometricBenchmark("test_conf.json")
        self.assertTrue(bench.config.dummy == 0)
        os.remove("test_conf.json")

    def test_save_conf(self):
        bench = GeometricBenchmark()
        bench.save_config("test_conf.json")
        with open("test_conf.json", "r") as fp:
            recovered = json.load(fp)
        for k in bench.config.keys():
            self.assertTrue(k in recovered.keys())
        os.remove("test_conf.json")

    def test_read_instance_set(self):
        bench = self.load_bench(DEFAULTS_STATIC)
        bench.read_instance_set()
        self.assertTrue(len(bench.config.instance_set.keys()) == 100)
        self.assertTrue(len(bench.config.instance_set[0]) == 7)
        self.assertTrue(len(bench.config.instance_set[0][0][1]) == "sigmoid")

        first_inst = bench.config.instance_set[0]

        bench2 = self.load_bench(DEFAULTS_STATIC)
        env = bench2.get_environment()
        self.assertTrue(env.instance_set[0] == first_inst)
        self.assertTrue(len(env.instance_set.keys()) == 100)
        self.assertTrue(len(bench.config.instance_set[0][0][1]) == "linear")

    def test_benchmark_env(self):
        bench = GeometricBenchmark()
        env = bench.get_benchmark()
        self.assertTrue(issubclass(type(env), GeometricEnv))

    def test_set_action_values_static(self):
        bench = self.load_bench(DEFAULTS_STATIC)
        bench.read_instance_set()
        bench._set_action_values()
        self.assertTrue(bench.config.action_value_default == 4)
        self.assertTrue(bench.config.action_values[0] == 4)
        self.assertTrue(bench.config.action_space_args == [16384])
        self.assertTrue(len(bench.config.observation_space_args[0]) == 29)
        self.assertTrue(
            len(bench.config.action_interval_mapping)
            == len(bench.config.action_value_mapping)
        )

    def test_set_action_values_dynamic(self):
        bench = self.load_bench(DEFAULTS_DYNAMIC)
        bench.read_instance_set()
        bench._set_action_values()
        self.assertTrue(bench.config.action_values_variable == True)
        self.assertTrue(bench.config.action_values[4] == 11)
        self.assertTrue(bench.config.action_space_args == [55440])
        self.assertTrue(len(bench.config.observation_space_args[0]) == 38)
