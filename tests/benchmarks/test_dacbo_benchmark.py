from __future__ import annotations

import json
import os
import unittest

pytest = __import__("pytest")
dacboenv = pytest.importorskip("dacboenv")

from dacbench.benchmarks import DACBOBenchmark
from dacbench.envs import DACBOEnv


class TestDACBOBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = DACBOBenchmark()
        bench.config["instance_set"] = {0: (1, "bbob/2/1/0")}
        bench.config["task_ids"] = ["bbob/2/1/0"]
        bench.config["inner_seeds"] = [1]
        bench.config["evaluation_mode"] = True
        env = bench.get_environment()
        assert issubclass(type(env), DACBOEnv)

    def test_setup(self):
        bench = DACBOBenchmark()
        expected_keys = [
            "observation_keys",
            "reward_keys",
            "action_space_class",
            "action_space_kwargs",
            "optimizer_cfg",
            "instance_set_path",
            "reward_range",
            "seed",
            "benchmark_info",
        ]
        for key in expected_keys:
            assert key in bench.config, f"Missing key: {key}"

    def test_read_instances(self):
        bench = DACBOBenchmark()
        bench.read_instance_set()
        # 20 tasks x 3 seeds = 60 instances
        assert len(bench.config.instance_set) == 60
        first = bench.config.instance_set[0]
        assert isinstance(first, tuple)
        assert len(first) == 2
        seed, task_id = first
        assert isinstance(seed, int)
        assert isinstance(task_id, str)

    def test_save_conf(self):
        bench = DACBOBenchmark()
        bench.read_instance_set()
        # Remove non-serializable keys (OmegaConf DictConfig, class refs, tuples)
        for key in [
            "config_space",
            "action_space_class",
            "optimizer_cfg",
            "instance_set",
        ]:
            if key in bench.config:
                del bench.config[key]
        bench.save_config("test_dacbo_conf.json")
        with open("test_dacbo_conf.json") as fp:
            recovered = json.load(fp)
        for k in bench.config:
            assert k in recovered
        os.remove("test_dacbo_conf.json")

    def test_from_to_json(self):
        bench = DACBOBenchmark()
        # Remove non-serializable keys before round-trip
        for key in ["optimizer_cfg", "action_space_class"]:
            if key in bench.config:
                del bench.config[key]
        json_str = bench.to_json()
        parsed = json.loads(json_str)
        assert parsed["instance_set_path"] == "bbob_2_default.yaml"
        assert parsed["seed"] == 0
        restored_bench = DACBOBenchmark.from_json(json_str)
        assert restored_bench.config["instance_set_path"] == "bbob_2_default.yaml"
        assert restored_bench.config["seed"] == 0
