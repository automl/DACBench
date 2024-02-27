from __future__ import annotations

import json
import os
import unittest

from dacbench.benchmarks import SGDBenchmark
from dacbench.envs import SGDEnv


class TestSGDBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = SGDBenchmark()
        env = bench.get_environment()
        assert issubclass(type(env), SGDEnv)

    def test_setup(self):
        bench = SGDBenchmark()
        assert bench.config is not None

        config = {"dummy": 0}
        with open("test_conf.json", "w+") as fp:
            json.dump(config, fp)
        bench = SGDBenchmark("test_conf.json")
        assert bench.config.dummy == 0
        os.remove("test_conf.json")

    def test_save_conf(self):
        bench = SGDBenchmark()
        del bench.config["config_space"]
        bench.save_config("test_conf.json")
        with open("test_conf.json") as fp:
            recovered = json.load(fp)
        for k in bench.config:
            assert k in recovered
        os.remove("test_conf.json")

    def test_read_instances(self):
        bench = SGDBenchmark()
        bench.read_instance_set()
        assert len(bench.config.instance_set.keys()) == 100
        inst = bench.config.instance_set[0]
        bench2 = SGDBenchmark()
        env = bench2.get_environment()
        assert len(env.instance_set.keys()) == 100
        # [3] instance architecture constructor functionally identical but not comparable
        assert inst[0] == env.instance_set[0][0]
        assert inst[1] == env.instance_set[0][1]

    def test_benchmark_env(self):
        bench = SGDBenchmark()
        env = bench.get_benchmark()
        assert issubclass(type(env), SGDEnv)

    def test_from_to_json(self):
        bench = SGDBenchmark()
        restored_bench = SGDBenchmark.from_json(bench.to_json())
        assert bench == restored_bench
