from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
from dacbench import AbstractEnv
from dacbench.wrappers import ObservationWrapper

dummy_config = {
    "instance_set": {0: 1},
    "benchmark_info": None,
    "cutoff": 10,
    "observation_space": gym.spaces.Dict(
        {
            "one": gym.spaces.Discrete(2),
            "two": gym.spaces.Box(low=np.array([-1, 1]), high=np.array([1, 5])),
        }
    ),
    "reward_range": (0, 1),
    "action_space": gym.spaces.Discrete(2),
}


class DummyDictEnv(AbstractEnv):
    def step(self, _):
        return {"one": 1, "two": np.array([1, 2])}, 0, False, False, {}

    def reset(self):
        return {}, {}


class TestObservationTrackingWrapper(unittest.TestCase):
    def get_test_env(self) -> AbstractEnv:
        return DummyDictEnv(dummy_config)

    def test_flatten(self):
        wrapped_env = ObservationWrapper(self.get_test_env())

        d = {"b": 0, "a": np.array([0, 1.4, 3])}
        flat = wrapped_env.flatten(d)

        expected = np.array([0, 1.4, 3, 0])

        np.testing.assert_array_almost_equal(flat, expected)

    def test_conversion_wrapper(self):
        action = 0.2

        env = self.get_test_env()
        reset_state_env, info = env.reset()
        step_state_env, *rest_env = env.step(action)
        assert isinstance(reset_state_env, dict)
        assert issubclass(type(info), dict)

        wrapped_env = ObservationWrapper(self.get_test_env())
        reset_state_wrapped, info = wrapped_env.reset()
        step_state_wrapped, *rest_wrapped = wrapped_env.step(action)

        assert isinstance(reset_state_wrapped, np.ndarray)
        self.assertListEqual(rest_env[1:], rest_wrapped[1:])

        np.testing.assert_array_equal(
            wrapped_env.flatten(reset_state_env).shape, reset_state_wrapped.shape
        )
        np.testing.assert_array_equal(
            wrapped_env.flatten(step_state_env).shape, step_state_wrapped.shape
        )
