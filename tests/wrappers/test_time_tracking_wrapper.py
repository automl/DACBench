import unittest

import numpy as np
from dacbench.benchmarks import LubyBenchmark
from dacbench.wrappers import EpisodeTimeWrapper


class TestTimeTrackingWrapper(unittest.TestCase):
    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = EpisodeTimeWrapper(env)
        self.assertTrue(len(wrapped.overall_times) == 0)
        self.assertTrue(wrapped.time_interval is None)
        wrapped.instance = [0]
        self.assertTrue(wrapped.instance[0] == 0)

        wrapped2 = EpisodeTimeWrapper(env, 10)
        self.assertTrue(len(wrapped2.overall_times) == 0)
        self.assertTrue(wrapped2.time_interval == 10)
        self.assertTrue(len(wrapped2.time_intervals) == 0)
        self.assertTrue(len(wrapped2.current_times) == 0)

    def test_step(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = EpisodeTimeWrapper(env, 10)

        state = wrapped.reset()
        self.assertTrue(len(state) > 1)

        state, reward, done, _ = wrapped.step(1)
        self.assertTrue(len(state) > 1)
        self.assertTrue(reward <= 0)
        self.assertFalse(done)

        self.assertTrue(len(wrapped.all_steps) == 1)
        self.assertTrue(len(wrapped.current_step_interval) == 1)
        self.assertTrue(len(wrapped.step_intervals) == 0)

        for _ in range(20):
            wrapped.step(1)

        self.assertTrue(len(wrapped.overall_times) > 2)
        self.assertTrue(len(wrapped.time_intervals) == 1)

    def test_get_times(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = EpisodeTimeWrapper(env)
        wrapped.reset()
        for i in range(5):
            wrapped.step(i)
        wrapped2 = EpisodeTimeWrapper(env, 2)
        wrapped2.reset()
        for i in range(5):
            wrapped2.step(i)

        overall_times_only, steps_only = wrapped.get_times()
        overall_times, steps, intervals, step_intervals = wrapped2.get_times()
        self.assertTrue(
            np.array_equal(
                np.round(overall_times, decimals=2),
                np.round(overall_times_only, decimals=2),
            )
        )
        self.assertTrue(len(step_intervals) == 3)
        self.assertTrue(len(step_intervals[0]) == 2)
        self.assertTrue(len(step_intervals[1]) == 2)
        self.assertTrue(len(step_intervals[2]) == 1)

    def test_rendering(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = EpisodeTimeWrapper(env, 10)
        wrapped.reset()
        for _ in range(30):
            wrapped.step(1)
        img = wrapped.render_step_time()
        self.assertTrue(img.shape[-1] == 3)
        img = wrapped.render_episode_time()
        self.assertTrue(img.shape[-1] == 3)
