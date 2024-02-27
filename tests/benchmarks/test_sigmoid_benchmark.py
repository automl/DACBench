from __future__ import annotations

import json
import os
import unittest

from dacbench.benchmarks import SigmoidBenchmark
from dacbench.envs import SigmoidEnv


class TestSigmoidBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = SigmoidBenchmark()
        env = bench.get_environment()
        assert issubclass(type(env), SigmoidEnv)

    def test_scenarios(self):
        scenarios = [
            "sigmoid_1D3M.json",
            "sigmoid_2D3M.json",
            "sigmoid_3D3M.json",
            "sigmoid_5D3M.json",
        ]
        for s in scenarios:
            path = os.path.join("dacbench/additional_configs/sigmoid", s)
            bench = SigmoidBenchmark(path)
            assert bench.config is not None
            env = bench.get_environment()
            state, info = env.reset()
            assert state is not None
            assert info is not None
            state, _, _, _, _ = env.step(env.action_space.sample())
            assert state is not None

    def test_save_conf(self):
        bench = SigmoidBenchmark()
        del bench.config["config_space"]
        bench.save_config("test_conf.json")
        with open("test_conf.json") as fp:
            recovered = json.load(fp)
        for k in bench.config:
            assert k in recovered
        os.remove("test_conf.json")

    def test_from_to_json(self):
        bench = SigmoidBenchmark()
        restored_bench = SigmoidBenchmark.from_json(bench.to_json())
        assert bench == restored_bench

    def test_read_instances(self):
        bench = SigmoidBenchmark()
        bench.read_instance_set()
        assert len(bench.config.instance_set.keys()) == 300
        assert len(bench.config.instance_set[0]) == 4
        first_inst = bench.config.instance_set[0]

        bench2 = SigmoidBenchmark()
        env = bench2.get_environment()
        assert len(env.instance_set[0]) == 4
        assert env.instance_set[0] == first_inst
        assert len(env.instance_set.keys()) == 300
