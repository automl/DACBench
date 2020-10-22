import unittest

import numpy as np
from dacbench.benchmarks import LubyBenchmark
from dacbench.wrappers import PerformanceTrackingWrapper


class TestTimeTrackingWrapper(unittest.TestCase):
    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = PerformanceTrackingWrapper(env)
        self.assertTrue(len(wrapped.overall_performance) == 0)
        self.assertTrue(wrapped.performance_interval is None)
        wrapped.instance = [0]
        self.assertTrue(wrapped.instance[0] == 0)

        wrapped2 = PerformanceTrackingWrapper(env, 10)
        self.assertTrue(len(wrapped2.overall_performance) == 0)
        self.assertTrue(wrapped2.performance_interval == 10)
        self.assertTrue(len(wrapped2.performance_intervals) == 0)
        self.assertTrue(len(wrapped2.current_performance) == 0)

    def test_step(self):
        bench = LubyBenchmark()
        bench.config.instance_set = [[0, 0], [1, 1], [3, 4], [5, 6]]
        env = bench.get_benchmark_env()
        wrapped = PerformanceTrackingWrapper(env, 10)

        state = wrapped.reset()
        self.assertTrue(len(state) > 1)

        state, reward, done, _ = wrapped.step(1)
        self.assertTrue(len(state) > 1)
        self.assertTrue(reward <= 0)
        self.assertFalse(done)

        while not done:
            _, _, done, _ = wrapped.step(1)

        self.assertTrue(len(wrapped.overall_performance) == 1)
        self.assertTrue(len(wrapped.performance_intervals) == 0)
        self.assertTrue(len(wrapped.current_performance) == 1)

        self.assertTrue(len(wrapped.instance_performances.keys()) == 1)
        wrapped.reset()
        done = False
        while not done:
            _, _, done, _ = wrapped.step(1)
        wrapped.reset()
        done = False
        while not done:
            _, _, done, _ = wrapped.step(1)
        self.assertTrue(len(wrapped.instance_performances.keys()) == 3)

    def test_get_performance(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = PerformanceTrackingWrapper(env)
        wrapped.reset()
        done = False
        while not done:
            _, _, done, _ = wrapped.step(1)
        wrapped2 = PerformanceTrackingWrapper(env, 2, track_instance_performance=False)
        wrapped2.reset()
        for i in range(5):
            wrapped2.step(i)
        wrapped3 = PerformanceTrackingWrapper(env, track_instance_performance=False)
        wrapped3.reset()
        for i in range(5):
            wrapped3.step(i)

        overall, instance_performance = wrapped.get_performance()
        overall_performance_only = wrapped3.get_performance()
        overall_performance, intervals = wrapped2.get_performance()
        self.assertTrue(
            np.array_equal(
                np.round(overall_performance, decimals=2),
                np.round(overall_performance_only, decimals=2),
            )
        )
       
        self.assertTrue(len(instance_performance.keys()) == 1)
        self.assertTrue(len(list(instance_performance.values())[0]) == 1)

        self.assertTrue(len(intervals) == 1)
        self.assertTrue(len(intervals[0]) == 0)
