import unittest

import numpy as np

from dacbench import AbstractEnv
from dacbench.benchmarks.cma_benchmark import CMAES_DEFAULTS, CMAESBenchmark


class TestCMAEnv(unittest.TestCase):
    def make_env(self):
        bench = CMAESBenchmark()
        env = bench.get_environment()
        return env

    def test_setup(self):
        env = self.make_env()
        self.assertTrue(issubclass(type(env), AbstractEnv))
        self.assertTrue(env.fbest is None)
        self.assertTrue(env.solutions is None)
        self.assertTrue(env.b is None)
        self.assertFalse(env.get_state is None)
        self.assertTrue(env.history_len == CMAES_DEFAULTS["hist_length"])
        self.assertTrue(env.popsize == CMAES_DEFAULTS["popsize"])

    def test_reset(self):
        env = self.make_env()
        env.reset()
        self.assertFalse(env.fcn is None)
        self.assertFalse(env.dim is None)
        self.assertFalse(env.init_sigma is None)
        self.assertFalse(env.cur_loc is None)
        self.assertFalse(env.es is None)

    def test_step(self):
        env = self.make_env()
        env.reset()
        _, reward, terminated, truncated, meta = env.step([1])
        self.assertTrue(reward >= env.reward_range[0])
        print(reward)
        self.assertTrue(reward <= env.reward_range[1])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(len(meta.keys()) == 0)

    def test_get_default_state(self):
        env = self.make_env()
        state, info = env.reset()
        self.assertTrue(issubclass(type(state), dict))
        self.assertTrue(issubclass(type(info), dict))
        self.assertTrue(
            np.array_equal(
                list(state.keys()),
                [
                    "current_loc",
                    "past_deltas",
                    "current_ps",
                    "current_sigma",
                    "history_deltas",
                    "past_sigma_deltas",
                ],
            )
        )
        self.assertTrue(len(state["current_ps"]) == 1)
        self.assertTrue(len(state["current_sigma"]) == 1)
        self.assertTrue(len(state["current_loc"]) == 10)
        self.assertTrue(len(state["past_deltas"]) == env.history_len)
        self.assertTrue(len(state["past_sigma_deltas"]) == env.history_len)
        self.assertTrue(len(state["history_deltas"]) == 2 * env.history_len)

        env.step([1])
        state, _, _, _, _ = env.step([1])
        self.assertTrue(issubclass(type(state), dict))
        self.assertTrue(
            np.array_equal(
                list(state.keys()),
                [
                    "current_loc",
                    "past_deltas",
                    "current_ps",
                    "current_sigma",
                    "history_deltas",
                    "past_sigma_deltas",
                ],
            )
        )
        self.assertTrue(len(state["current_ps"]) == 1)
        self.assertTrue(len(state["current_sigma"]) == 1)
        self.assertTrue(len(state["current_loc"]) == 10)
        self.assertTrue(len(state["past_deltas"]) == env.history_len)
        self.assertTrue(len(state["past_sigma_deltas"]) == env.history_len)
        self.assertTrue(len(state["history_deltas"]) == 2 * env.history_len)

    def test_close(self):
        env = self.make_env()
        self.assertTrue(env.close())

    def test_render(self):
        env = self.make_env()
        env.render("human")
        with self.assertRaises(NotImplementedError):
            env.render("random")
