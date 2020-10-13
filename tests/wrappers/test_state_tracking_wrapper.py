import pytest
import unittest

import numpy as np
from gym import spaces
from daclib.benchmarks import LubyBenchmark
from daclib.wrappers import StateTrackingWrapper


class TestStateTrackingWrapper(unittest.TestCase):
    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = StateTrackingWrapper(env)
        self.assertTrue(len(wrapped.overall)==0)
        self.assertTrue(wrapped.tracking_interval is None)
        wrapped.instance = [0]
        self.assertTrue(wrapped.instance[0]==0)

        wrapped2 = StateTrackingWrapper(env, 10)
        self.assertTrue(len(wrapped2.overall)==0)
        self.assertTrue(wrapped2.tracking_interval==10)
        self.assertTrue(len(wrapped2.interval_list)==0)
        self.assertTrue(len(wrapped2.current_interval)==0)

    def test_step_reset(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = StateTrackingWrapper(env, 10)

        state = wrapped.reset()
        self.assertTrue(len(state)>1)
        self.assertTrue(len(wrapped.overall)==1)

        state, reward, done, _ = wrapped.step(1)
        self.assertTrue(len(state)>1)
        self.assertTrue(reward < 0)
        self.assertFalse(done)

        self.assertTrue(len(wrapped.overall)==2)
        self.assertTrue(len(wrapped.current_interval)==2)
        self.assertTrue(len(wrapped.interval_list)==0)


    def test_get_states(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = StateTrackingWrapper(env)
        wrapped.reset()
        for i in range(4):
            wrapped.step(i)
        wrapped2 = StateTrackingWrapper(env, 2)
        wrapped2.reset()
        for i in range(4):
            wrapped2.step(i)

        overall_only = wrapped.get_states()
        overall, intervals = wrapped2.get_states()
        self.assertTrue(np.array_equal(overall, overall_only))
        self.assertTrue(len(overall_only)==5)
        self.assertTrue(len(overall_only[4])==21)

        print(intervals)
        self.assertTrue(len(intervals)==3)
        self.assertTrue(len(intervals[0])==2)
        self.assertTrue(len(intervals[1])==2)
        self.assertTrue(len(intervals[2])==1)

    # TODO
    def test_rendering(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = StateTrackingWrapper(env, 10)
