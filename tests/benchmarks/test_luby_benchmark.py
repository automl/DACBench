from __future__ import annotations

import json
import os
import unittest

import numpy as np
from dacbench.benchmarks import LubyBenchmark
from dacbench.envs import LubyEnv
from dacbench.wrappers import RewardNoiseWrapper


class TestLubyBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
        assert issubclass(type(env), LubyEnv)

    def test_scenarios(self):
        scenarios = ["luby_hard.json", "luby_harder.json", "luby_very_hard.json"]
        for s in scenarios:
            path = os.path.join("dacbench/additional_configs/luby/", s)
            bench = LubyBenchmark(path)
            assert bench.config is not None
            env = bench.get_environment()
            state, info = env.reset()
            assert state is not None
            assert info is not None
            state, _, _, _, _ = env.step(0)
            assert state is not None

    def test_save_conf(self):
        bench = LubyBenchmark()
        del bench.config["config_space"]
        bench.save_config("test_conf.json")
        with open("test_conf.json") as fp:
            recovered = json.load(fp)
        for k in bench.config:
            assert k in recovered
        os.remove("test_conf.json")

    def test_read_instances(self):
        bench = LubyBenchmark()
        bench.read_instance_set()
        print(bench.config.instance_set)
        assert len(bench.config.instance_set.keys()) == 1
        assert len(bench.config.instance_set[0]) == 2
        assert bench.config.instance_set[0] == [0, 0]
        bench2 = LubyBenchmark()
        env = bench2.get_environment()
        assert len(env.instance_set[0]) == 2
        assert env.instance_set[0] == [0, 0]
        assert len(env.instance_set.keys()) == 1

    def test_benchmark_env(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark()
        assert issubclass(type(env), RewardNoiseWrapper)
        env.reset()
        _, r, _, _, _ = env.step(1)
        assert r != 0
        assert r != -1

    def test_cutoff_setting(self):
        bench = LubyBenchmark()
        bench.set_cutoff(100)
        assert bench.config.cutoff == 100
        assert bench.config.action_space_args == [int(np.log2(100))]

    def test_history_len_setting(self):
        bench = LubyBenchmark()
        bench.set_history_length(20)
        assert len(bench.config.observation_space_args[0]) == 21

    def test_from_to_json(self):
        bench = LubyBenchmark()
        restored_bench = LubyBenchmark.from_json(bench.to_json())
        assert bench == restored_bench
