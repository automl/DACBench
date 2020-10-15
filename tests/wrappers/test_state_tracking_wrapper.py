import unittest
import pytest

import gym
import numpy as np
from daclib.benchmarks import LubyBenchmark, CMAESBenchmark
from daclib.wrappers import StateTrackingWrapper


class TestStateTrackingWrapper(unittest.TestCase):
    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = StateTrackingWrapper(env)
        self.assertTrue(len(wrapped.overall) == 0)
        self.assertTrue(wrapped.tracking_interval is None)
        wrapped.instance = [0]
        self.assertTrue(wrapped.instance[0] == 0)

        wrapped2 = StateTrackingWrapper(env, 10)
        self.assertTrue(len(wrapped2.overall) == 0)
        self.assertTrue(wrapped2.tracking_interval == 10)
        self.assertTrue(len(wrapped2.interval_list) == 0)
        self.assertTrue(len(wrapped2.current_interval) == 0)

    def test_step_reset(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = StateTrackingWrapper(env, 2)

        state = wrapped.reset()
        self.assertTrue(len(state) > 1)
        self.assertTrue(len(wrapped.overall) == 1)

        state, reward, done, _ = wrapped.step(1)
        self.assertTrue(len(state) > 1)
        self.assertTrue(reward < 0)
        self.assertFalse(done)

        self.assertTrue(len(wrapped.overall) == 2)
        self.assertTrue(len(wrapped.current_interval) == 2)
        self.assertTrue(len(wrapped.interval_list) == 0)

        state = wrapped.reset()
        self.assertTrue(len(wrapped.overall) == 3)
        self.assertTrue(len(wrapped.current_interval) == 1)
        self.assertTrue(len(wrapped.interval_list) == 1)

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
        self.assertTrue(len(overall_only) == 5)
        self.assertTrue(len(overall_only[4]) == 6)

        self.assertTrue(len(intervals) == 3)
        self.assertTrue(len(intervals[0]) == 2)
        self.assertTrue(len(intervals[1]) == 2)
        self.assertTrue(len(intervals[2]) == 1)

    def test_rendering(self):
        bench = CMAESBenchmark()
        env = bench.get_benchmark_env()
        wrapped = StateTrackingWrapper(env)
        wrapped.reset()
        with pytest.raises(NotImplementedError):
            wrapped.render_state_tracking()

        bench = CMAESBenchmark()
        def dummy():
            return [1, [2, 3]]
        bench.config.state_method = dummy
        bench.config.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Box(low=np.array([-1, 1]), high=np.array([5, 5]))))
        env = bench.get_benchmark_env()
        wrapped = StateTrackingWrapper(env)
        wrapped.reset()
        with pytest.raises(NotImplementedError):
            wrapped.render_state_tracking()

        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = StateTrackingWrapper(env, 2)
        wrapped.reset()
        wrapped.step(1)
        wrapped.step(1)
        img = wrapped.render_state_tracking()
        self.assertTrue(img.shape[-1]==3)

        class discrete_obs_env():
            def __init__(self):
                self.observation_space = gym.spaces.Discrete(2)
                self.action_space = gym.spaces.Discrete(2)
                self.reward_range = (1, 2)
                self.metadata = {}
            def reset(self):
                return 1

            def step(self, action):
                return 1, 1, 1, 1
        env = discrete_obs_env()
        wrapped = StateTrackingWrapper(env)
        wrapped.reset()
        wrapped.step(1)
        img = wrapped.render_state_tracking()
        self.assertTrue(img.shape[-1]==3)

        class multi_discrete_obs_env():
            def __init__(self):
                self.observation_space = gym.spaces.MultiDiscrete([2, 3])
                self.action_space = gym.spaces.Discrete(2)
                self.reward_range = (1, 2)
                self.metadata = {}
            def reset(self):
                return [1, 2]

            def step(self, action):
                return [1, 2], 1, 1, 1
        env = multi_discrete_obs_env()
        wrapped = StateTrackingWrapper(env)
        wrapped.reset()
        wrapped.step(1)
        img = wrapped.render_state_tracking()
        self.assertTrue(img.shape[-1]==3)

        class multi_binary_obs_env():
            def __init__(self):
                self.observation_space = gym.spaces.MultiBinary(2)
                self.action_space = gym.spaces.Discrete(2)
                self.reward_range = (1, 2)
                self.metadata = {}
            def reset(self):
                return [1, 1]

            def step(self, action):
                return [1, 1], 1, 1, 1
        env = multi_binary_obs_env()
        wrapped = StateTrackingWrapper(env)
        wrapped.reset()
        wrapped.step(1)
        img = wrapped.render_state_tracking()
        self.assertTrue(img.shape[-1]==3)
