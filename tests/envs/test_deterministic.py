from __future__ import annotations

import unittest

import numpy as np
from dacbench import benchmarks
from numpy.testing import assert_almost_equal


def assert_state_space_equal(state1, state2):
    assert isinstance(state1, type(state2))

    if isinstance(state1, np.ndarray):
        assert_almost_equal(state1, state2)
    elif isinstance(state1, dict):
        assert state1.keys() == state2.keys()
        for key in state1:
            if "history" not in key:
                assert_almost_equal(state1[key], state2[key])
    else:
        raise NotImplementedError(f"State space type {type(state1)} not comparable")


class TestDeterministic(unittest.TestCase):
    def run_deterministic_test(self, benchmark_name, seed=42):
        print(benchmark_name)
        bench = getattr(benchmarks, benchmark_name)()
        action = bench.get_environment().action_space.sample()

        env1 = bench.get_benchmark(seed=seed)
        init_state1, info1 = env1.reset()
        _, reward1, terminated1, truncated1, info1 = env1.step(action)

        env2 = bench.get_benchmark(seed=seed)
        init_state2, info2 = env2.reset()
        _, reward2, terminated2, truncated2, info2 = env2.step(action)

        assert_state_space_equal(init_state1, init_state2)
        assert info1 == info2
        assert terminated1 == terminated2
        assert truncated1 == truncated2
        assert info1 == info2

    def test_LubyBenchmark(self):
        self.run_deterministic_test("LubyBenchmark")

    def test_SigmoidBenchmark(self):
        self.run_deterministic_test("SigmoidBenchmark")

    # FD Test are hard to run due to old FD version
    # def test_FastDownwardBenchmark(self):
    #     benchmark_name = "FastDownwardBenchmark"
    #     seed = 42
    #     bench = getattr(benchmarks, benchmark_name)()
    #     action = bench.get_environment().action_space.sample()

    #     env1 = bench.get_benchmark(seed=seed)
    #     init_state1, info1 = env1.reset()
    #     state1, reward1, terminated1, truncated1, info1 = env1.step(action)
    #     env1.close()

    #     env2 = bench.get_benchmark(seed=seed)
    #     init_state2, info2 = env2.reset()
    #     state2, reward2, terminated2, truncated2, info2 = env2.step(action)
    #     env2.close()

    #     assert_state_space_equal(init_state1, init_state2)
    #     assert_state_space_equal(state1, state2)
    #     self.assertEqual(reward1, reward2)
    #     self.assertEqual(info1, info2)
    #     self.assertEqual(terminated1, terminated2)
    #     self.assertEqual(truncated1, truncated2)
    #     self.assertEqual(info1, info2)

    def test_SGDBenchmark(self):
        self.run_deterministic_test("SGDBenchmark")

    def test_OneLLBenchmark(self):
        ...
        # todo
        # self.run_deterministic_test("OneLLBenchmark")
