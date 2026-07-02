from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import numpy as np

from tests.helpers import tiny_sgd_loader
from dacbench.benchmarks import SGDBenchmark
from dacbench.envs import SGDEnv, SGDInstance


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
        assert len(bench.config.instance_set.keys()) == 10
        inst = bench.config.instance_set[0]
        bench2 = SGDBenchmark()
        env = bench2.get_environment()
        assert len(env.instance_set.keys()) == 10
        # [3] instance architecture constructor functionally identical but not comparable
        assert isinstance(env.instance_set[0], SGDInstance)
        assert inst.optimizer_type == env.instance_set[0].optimizer_type
        assert inst.dataset_path == env.instance_set[0].dataset_path
        assert inst.dataset_name == env.instance_set[0].dataset_name
        assert inst.batch_size == env.instance_set[0].batch_size
        assert inst.fraction_of_dataset == env.instance_set[0].fraction_of_dataset
        assert inst.train_validation_ratio == env.instance_set[0].train_validation_ratio
        assert inst.seed == env.instance_set[0].seed

    def test_benchmark_env(self):
        bench = SGDBenchmark()
        env = bench.get_benchmark()
        assert issubclass(type(env), SGDEnv)

    def test_from_to_json(self):
        bench = SGDBenchmark()
        restored_bench = SGDBenchmark.from_json(bench.to_json())
        assert bench.config.keys() == restored_bench.config.keys(), f"Configs should have same keys"
        for k in bench.config.keys():
            if k in ["reward_range", "observation_space_kwargs", "torch_hub_model"]:
                assert np.allclose(bench.config[k], restored_bench.config[k]), f"Config values should be equal, got: {bench.config[k]} != {restored_bench.config[k]}"
            elif k == "layer_specification":
                for layer in bench.config[k]:
                    assert layer in restored_bench.config[k], f"Layer {layer} should be in restored config"
            elif k == "optimizer_params":
                for kk in bench.config[k].keys():
                    assert np.allclose(bench.config[k][kk], restored_bench.config[k][kk]), f"Config values should be equal, got: {bench.config[k][kk]} != {restored_bench.config[k][kk]}"
            else:
                assert bench.config[k] == restored_bench.config[k], f"Config values should be equal, got: {bench.config[k]} != {restored_bench.config[k]}"

    def test_seeding(self):
        bench = SGDBenchmark()
        bench.config.seed = 123
        bench.config.seed_action_space = True
        bench.config.instance_update_func = "no_progression"
        mems = []

        with patch("dacbench.envs.sgd.random_torchvision_loader", side_effect=tiny_sgd_loader):
            for _ in range(2):
                env = bench.get_environment()
                assert env.instance is env.instance_set[0]

                state, _ = env.reset()

                terminated, truncated = False, False
                mem = []
                step = 0
                while not (terminated or truncated) and step < 5:
                    action = env.action_space.sample()
                    state, reward, terminated, truncated, _ = env.step(action)
                    mem.append([state, [reward, int(truncated), action]])
                    step += 1
                mems.append(mem)

        # Trajectories must be reproducible under the same seed.
        assert len(mems[0]) == len(mems[1])
        for mem0, mem1 in zip(mems[0], mems[1], strict=True):
            state0, (reward0, trunc0, action0) = mem0
            state1, (reward1, trunc1, action1) = mem1
            assert trunc0 == trunc1
            assert np.allclose(reward0, reward1), f"reward differs: {reward0} vs {reward1}"
            assert np.allclose(action0, action1), f"action differs: {action0} vs {action1}"
            assert state0.keys() == state1.keys()
            for k in state0:
                v0, v1 = state0[k], state1[k]
                if isinstance(v0, (bool, int)):
                    assert v0 == v1, f"state[{k}] differs: {v0!r} vs {v1!r}"
                else:
                    assert np.allclose(v0, v1), f"state[{k}] differs: {v0!r} vs {v1!r}"
