from __future__ import annotations

import unittest

import numpy as np
import pytest
from dacbench.abstract_env import AbstractEnv
from gymnasium import spaces


class LessAbstractEnv(AbstractEnv):
    def reset(self, seed: int | None = None):
        pass

    def step(self, action):
        pass


class TestAbstractEnv(unittest.TestCase):
    def test_exceptions(self):
        config = {
            "action_space_class": "Discrete",
            "action_space_args": [4],
            "observation_space_class": "Dict",
            "observation_space_type": np.float32,
            "observation_space_args": [
                np.array([-1, -1, -1], dtype=np.float32),
                np.array([1, 1, 1], dtype=np.float32),
            ],
            "reward_range": (-1, 0),
            "cutoff": 30,
            "instance_set": {0: 1, 1: 1},
            "benchmark_info": None,
        }
        with pytest.raises(TypeError):
            LessAbstractEnv(config)

        config = {
            "action_space_class": "Discrete",
            "action_space_args": [4],
            "observation_space_class": "Box",
            "observation_space_type": np.float32,
            "reward_range": (-1, 0),
            "cutoff": 30,
            "benchmark_info": None,
            "instance_set": {0: 1, 1: 1},
        }
        with pytest.raises(KeyError):
            LessAbstractEnv(config)

        config = {
            "action_space_class": "Discrete",
            "action_space_args": [4],
            "observation_space_type": np.float32,
            "observation_space_args": [
                np.array([-1, -1, -1], dtype=np.float32),
                np.array([1, 1, 1], dtype=np.float32),
            ],
            "reward_range": (-1, 0),
            "cutoff": 30,
            "benchmark_info": None,
            "instance_set": {0: 1, 1: 1},
        }
        with pytest.raises(KeyError):
            LessAbstractEnv(config)

        config = {
            "action_space_class": "Tuple",
            "action_space_args": np.array([4]).astype(np.float32),
            "observation_space_class": "Box",
            "observation_space_type": np.float32,
            "observation_space_args": [
                np.array([-1, -1, -1], dtype=np.float32),
                np.array([1, 1, 1], dtype=np.float32),
            ],
            "reward_range": (-1, 0),
            "cutoff": 30,
            "benchmark_info": None,
            "instance_set": {0: 1, 1: 1},
        }
        with pytest.raises(TypeError):
            LessAbstractEnv(config)

        config = {
            "action_space_args": np.array([4]).astype(np.float32),
            "observation_space_class": "Box",
            "observation_space_type": np.float32,
            "observation_space_args": [
                np.array([-1, -1, -1], dtype=np.float32),
                np.array([1, 1, 1], dtype=np.float32),
            ],
            "reward_range": (-1, 0),
            "cutoff": 30,
            "benchmark_info": None,
            "instance_set": {0: 1, 1: 1},
        }
        with pytest.raises(KeyError):
            LessAbstractEnv(config)

    def make_env(self):
        config = {
            "action_space_class": "Discrete",
            "action_space_args": [4],
            "observation_space_class": "Box",
            "observation_space_type": np.float32,
            "observation_space_args": [
                np.array([-1, -1, -1], dtype=np.float32),
                np.array([1, 1, 1], dtype=np.float32),
            ],
            "reward_range": (-1, 0),
            "cutoff": 30,
            "benchmark_info": None,
            "instance_set": {0: 1, 1: 1},
        }
        return LessAbstractEnv(config)

    def test_setup(self):
        env = self.make_env()
        assert len(env.instance_set) >= 1
        assert env.n_steps > 0
        assert type(env.reward_range) is tuple
        assert issubclass(type(env.observation_space), spaces.Space)
        assert issubclass(type(env.action_space), spaces.Space)

        config = {
            "action_space": spaces.Discrete(2),
            "observation_space": spaces.Discrete(2),
            "reward_range": (-1, 0),
            "cutoff": 30,
            "benchmark_info": None,
            "instance_set": {0: 1, 1: 1},
        }
        env = LessAbstractEnv(config)
        assert len(env.instance_set) >= 1
        assert env.n_steps > 0
        assert type(env.reward_range) is tuple
        assert issubclass(type(env.observation_space), spaces.Space)
        assert issubclass(type(env.action_space), spaces.Space)

    def test_pre_step_and_reset(self):
        env = self.make_env()

        env.n_steps = 10
        assert not env.step_()
        env.n_steps = 1
        assert env.step_()

        env.inst_id = 0
        env.reset_()
        assert env.inst_id == 1
        assert env.c_step == 0

    def test_getters_and_setters(self):
        env = self.make_env()

        assert env.inst_id == env.get_inst_id()
        env.set_inst_id(1)
        assert env.get_inst_id() == 1

        assert env.instance == env.get_instance()
        env.set_instance(100)
        assert env.get_instance() == 100

        assert all(
            env.instance_set[k] == env.get_instance_set()[k]
            for k in range(len(env.instance_set))
        )
        env.set_instance_set({0: 100})
        assert env.get_instance_set()[0] == 100

    def test_seed(self):
        env = self.make_env()
        seeds = []
        for _ in range(10):
            seeds.append(env.seed()[0])
        assert not len(set(seeds)) < 8
