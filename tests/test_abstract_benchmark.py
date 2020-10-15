import pytest
import unittest
import json
import os
from daclib.abstract_benchmark import AbstractBenchmark, objdict


class TestAbstractBenchmark(unittest.TestCase):
    def test_not_implemented_method(self):
        bench = AbstractBenchmark()
        with pytest.raises(NotImplementedError):
            bench.get_benchmark_env()

    def test_setup(self):
        bench = AbstractBenchmark()
        self.assertTrue(bench.config is None)

    def test_config_file_management(self):
        bench = AbstractBenchmark()

        bench.config = objdict({"seed": 0})
        test_config = objdict({"seed": 10})
        with open("test_conf.json", "w+") as fp:
            json.dump(test_config, fp)
        self.assertTrue(bench.config.seed == 0)
        bench.read_config_file("test_conf.json")
        self.assertTrue(bench.config.seed == 10)
        self.assertTrue(len(bench.config.keys()) == 1)
        os.remove("test_conf.json")

        bench.save_config("test_conf2.json")
        with open("test_conf2.json", "r") as fp:
            recovered = json.load(fp)
        self.assertTrue(recovered["seed"] == 10)
        self.assertTrue(len(recovered.keys()) == 1)
        os.remove("test_conf2.json")

    def test_attributes(self):
        bench = AbstractBenchmark()
        bench.config = objdict({"seed": 0})
        self.assertTrue(bench.config.seed == bench.config["seed"])
        bench.config.seed = 42
        self.assertTrue(bench.config["seed"] == 42)

    def test_getters_and_setters(self):
        bench = AbstractBenchmark()
        bench.config = objdict({"seed": 0})
        config = bench.get_config()
        self.assertTrue(issubclass(type(config), dict))

        bench.set_seed(100)
        self.assertTrue(bench.config.seed == 100)

        bench.set_action_space("Discrete", [4])
        self.assertTrue(bench.config.action_space == "Discrete")
        self.assertTrue(bench.config.action_space_args == [4])

        bench.set_observation_space("Box", [[1], [0]], float)
        self.assertTrue(bench.config.observation_space == "Box")
        self.assertTrue(bench.config.observation_space_args[0] == [1])
        self.assertTrue(bench.config.observation_space_type == float)

    def test_objdict(self):
        d = objdict({"dummy": 0})

        self.assertTrue(d["dummy"] == d.dummy)
        with pytest.raises(KeyError):
            d["error"]
        with pytest.raises(AttributeError):
            d.error

        d["error"] = 12
        self.assertTrue(d.error == 12)
        del d.error
        self.assertFalse("error" in d.keys())

        with pytest.raises(KeyError):
            del d["error"]
        with pytest.raises(AttributeError):
            del d.error
