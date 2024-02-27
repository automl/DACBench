from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from dacbench.benchmarks import LubyBenchmark
from dacbench.wrappers import PerformanceTrackingWrapper


class TestPerformanceWrapper(unittest.TestCase):
    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = PerformanceTrackingWrapper(env)
        assert len(wrapped.overall_performance) == 0
        assert wrapped.performance_interval is None
        wrapped.instance = [0]
        assert wrapped.instance[0] == 0

        wrapped2 = PerformanceTrackingWrapper(env, 10)
        assert len(wrapped2.overall_performance) == 0
        assert wrapped2.performance_interval == 10
        assert len(wrapped2.performance_intervals) == 0
        assert len(wrapped2.current_performance) == 0

    def test_step(self):
        bench = LubyBenchmark()
        bench.config.instance_set = {0: [0, 0], 1: [1, 1], 2: [3, 4], 3: [5, 6]}
        env = bench.get_environment()
        wrapped = PerformanceTrackingWrapper(env, 2)

        state, info = wrapped.reset()
        assert len(state) > 1
        assert issubclass(type(info), dict)

        state, reward, terminated, truncated, _ = wrapped.step(1)
        assert len(state) > 1
        assert reward <= 0
        assert not terminated
        assert not truncated

        while not (terminated or truncated):
            _, _, terminated, truncated, _ = wrapped.step(1)

        assert len(wrapped.overall_performance) == 1
        assert len(wrapped.performance_intervals) == 0
        assert len(wrapped.current_performance) == 1
        assert len(wrapped.instance_performances.keys()) == 1

        terminated, truncated = False, False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = wrapped.step(1)
        terminated, truncated = False, False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = wrapped.step(1)

        assert len(wrapped.performance_intervals) == 1
        assert len(wrapped.current_performance) == 1

        wrapped.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = wrapped.step(1)
        wrapped.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = wrapped.step(1)
        assert len(wrapped.instance_performances.keys()) == 3

        wrapped.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = wrapped.step(1)
        wrapped.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = wrapped.step(1)
        assert len(wrapped.instance_performances.keys()) == 4

    def test_get_performance(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = PerformanceTrackingWrapper(env)
        wrapped.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = wrapped.step(1)
        wrapped2 = PerformanceTrackingWrapper(env, 2)
        wrapped2.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = wrapped2.step(1)
        wrapped3 = PerformanceTrackingWrapper(env, 2, track_instance_performance=False)
        wrapped3.reset()
        for i in range(5):
            wrapped3.step(i)
        wrapped4 = PerformanceTrackingWrapper(env, track_instance_performance=False)
        wrapped4.reset()
        for i in range(5):
            wrapped4.step(i)

        overall, instance_performance = wrapped.get_performance()
        overall_perf, interval_perf, instance_perf = wrapped2.get_performance()
        overall_performance_only = wrapped4.get_performance()
        overall_performance, intervals = wrapped3.get_performance()
        assert np.array_equal(
            np.round(overall_performance, decimals=2),
            np.round(overall_performance_only, decimals=2),
        )

        assert np.array_equal(
            np.round(overall_perf, decimals=2), np.round(overall, decimals=2)
        )

        assert len(instance_performance.keys()) == 1
        assert len(next(iter(instance_performance.values()))) == 1
        assert len(instance_perf.keys()) == 1
        assert len(next(iter(instance_perf.values()))) == 1

        assert len(intervals) == 1
        assert len(intervals[0]) == 0
        assert len(interval_perf) == 1
        assert len(interval_perf[0]) == 1

    @mock.patch("dacbench.wrappers.performance_tracking_wrapper.plt")
    def test_render(self, mock_plt):
        bench = LubyBenchmark()
        env = bench.get_environment()
        env = PerformanceTrackingWrapper(env)
        for _ in range(10):
            terminated, truncated = False, False
            env.reset()
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(1)
        env.render_performance()
        assert mock_plt.show.called
        env.render_instance_performance()
        assert mock_plt.show.called
