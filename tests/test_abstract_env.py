import pytest
import unittest

import numpy as np
from gym import spaces
from daclib.abstract_env import AbstractEnv


class TestAbstractEnv(unittest.TestCase):
    def test_not_implemented_methods(self):
        env = self.make_env()
        with pytest.raises(NotImplementedError):
            env.step(0)

        with pytest.raises(NotImplementedError):
            env.reset()

    def make_env(self):
        config = {
            "action_space": "Discrete",
            "action_space_args": np.array([4]).astype(np.float32),
            "observation_space": "Box",
            "observation_space_type": np.float32,
            "observation_space_args": [
                np.array([-1, -1, -1], dtype=np.float32),
                np.array([1, 1, 1], dtype=np.float32),
            ],
            "reward_range": (-1, 0),
            "cutoff": 30,
            "instance_set": [[1], [1]],
        }
        env = AbstractEnv(config)
        return env

    def test_setup(self):
        env = self.make_env()
        self.assertTrue(len(env.instance_set) >= 1)
        self.assertTrue(env.n_steps > 0)
        self.assertTrue(type(env.reward_range) is tuple)
        print(type(env.observation_space))
        self.assertTrue(issubclass(type(env.observation_space), spaces.Space))
        self.assertTrue(issubclass(type(env.action_space), spaces.Space))

    def test_pre_step_and_reset(self):
        env = self.make_env()

        env.n_steps = 10
        self.assertFalse(env.step_())
        env.n_steps = 1
        self.assertTrue(env.step_())

        env.inst_id = 0
        env.reset_()
        self.assertTrue(env.inst_id == 1)
        self.assertTrue(env.c_step == 0)

    def test_getters_and_setters(self):
        env = self.make_env()

        self.assertTrue(env.inst_id == env.get_inst_id())
        env.set_inst_id(100)
        self.assertTrue(100 == env.get_inst_id())

        self.assertTrue(env.instance == env.get_instance())
        env.set_instance(100)
        self.assertTrue(100 == env.get_instance())

        self.assertTrue(
            all(
                [
                    env.instance_set[k] == env.get_instance_set()[k]
                    for k in range(len(env.instance_set))
                ]
            )
        )
        env.set_instance_set(100)
        self.assertTrue(100 == env.get_instance_set())
