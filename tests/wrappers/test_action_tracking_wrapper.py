import pytest
import unittest

import numpy as np
from gym import spaces
from daclib.benchmarks import LubyBenchmark
from daclib.wrappers import ActionFrequencyWrapper


class TestActionTrackingWrapper(unittest.TestCase):
    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = ActionFrequencyWrapper(env)
        self.assertTrue(len(wrapped.overall) == 0)
        self.assertTrue(wrapped.tracking_interval is None)
        wrapped.instance = [0]
        self.assertTrue(wrapped.instance[0] == 0)

        wrapped2 = ActionFrequencyWrapper(env, 10)
        self.assertTrue(len(wrapped2.overall) == 0)
        self.assertTrue(wrapped2.tracking_interval == 10)
        self.assertTrue(len(wrapped2.interval_list) == 0)
        self.assertTrue(len(wrapped2.current_interval) == 0)

    def test_step(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = ActionFrequencyWrapper(env, 10)

        state = wrapped.reset()
        self.assertTrue(len(state) > 1)

        state, reward, done, _ = wrapped.step(1)
        self.assertTrue(len(state) > 1)
        self.assertTrue(reward < 0)
        self.assertFalse(done)

        self.assertTrue(len(wrapped.overall) == 1)
        self.assertTrue(wrapped.overall[0] == 1)
        self.assertTrue(len(wrapped.current_interval) == 1)
        self.assertTrue(wrapped.current_interval[0] == 1)
        self.assertTrue(len(wrapped.interval_list) == 0)

    def test_get_actions(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = ActionFrequencyWrapper(env)
        wrapped.reset()
        for i in range(5):
            wrapped.step(i)
        wrapped2 = ActionFrequencyWrapper(env, 2)
        wrapped2.reset()
        for i in range(5):
            wrapped2.step(i)

        overall_only = wrapped.get_actions()
        overall, intervals = wrapped2.get_actions()
        self.assertTrue(np.array_equal(overall, overall_only))
        self.assertTrue(overall_only == [0, 1, 2, 3, 4])

        self.assertTrue(len(intervals) == 3)
        self.assertTrue(len(intervals[0]) == 2)
        self.assertTrue(intervals[0] == [0, 1])
        self.assertTrue(len(intervals[1]) == 2)
        self.assertTrue(intervals[1] == [2, 3])
        self.assertTrue(len(intervals[2]) == 1)
        self.assertTrue(intervals[2] == [4])

    # TODO
    def test_rendering(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = ActionFrequencyWrapper(env, 10)
