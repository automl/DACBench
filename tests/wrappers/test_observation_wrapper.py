import unittest

import numpy as np

from dacbench import AbstractEnv
from dacbench.benchmarks import CMAESBenchmark
from dacbench.wrappers import ObservationWrapper


class TestObservationTrackingWrapper(unittest.TestCase):
    def get_test_env(self) -> AbstractEnv:
        bench = CMAESBenchmark()
        env = bench.get_benchmark(seed=42)
        return env

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
        self.assertIsInstance(reset_state_env, dict)
        self.assertTrue(issubclass(type(info), dict))

        wrapped_env = ObservationWrapper(self.get_test_env())
        reset_state_wrapped, info = wrapped_env.reset()
        step_state_wrapped, *rest_wrapped = wrapped_env.step(action)

        self.assertIsInstance(reset_state_wrapped, np.ndarray)
        self.assertListEqual(rest_env[1:], rest_wrapped[1:])

        np.testing.assert_array_equal(
            wrapped_env.flatten(reset_state_env).shape, reset_state_wrapped.shape
        )
        np.testing.assert_array_equal(
            wrapped_env.flatten(step_state_env).shape, step_state_wrapped.shape
        )
