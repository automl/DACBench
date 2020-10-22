import unittest
import pytest
import gym
import numpy as np
from dacbench.benchmarks import LubyBenchmark, FastDownwardBenchmark, CMAESBenchmark
from dacbench.wrappers import ActionFrequencyWrapper


class TestActionTrackingWrapper(unittest.TestCase):
    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = ActionFrequencyWrapper(env)
        self.assertTrue(len(wrapped.overall_actions) == 0)
        self.assertTrue(wrapped.action_interval is None)
        wrapped.instance = [0]
        self.assertTrue(wrapped.instance[0] == 0)

        wrapped2 = ActionFrequencyWrapper(env, 10)
        self.assertTrue(len(wrapped2.overall_actions) == 0)
        self.assertTrue(wrapped2.action_interval == 10)
        self.assertTrue(len(wrapped2.action_intervals) == 0)
        self.assertTrue(len(wrapped2.current_actions) == 0)

    def test_step(self):
        bench = LubyBenchmark()
        env = bench.get_benchmark_env()
        wrapped = ActionFrequencyWrapper(env, 10)

        state = wrapped.reset()
        self.assertTrue(len(state) > 1)

        state, reward, done, _ = wrapped.step(1)
        self.assertTrue(len(state) > 1)
        self.assertTrue(reward <= 0)
        self.assertFalse(done)

        self.assertTrue(len(wrapped.overall_actions) == 1)
        self.assertTrue(wrapped.overall_actions[0] == 1)
        self.assertTrue(len(wrapped.current_actions) == 1)
        self.assertTrue(wrapped.current_actions[0] == 1)
        self.assertTrue(len(wrapped.action_intervals) == 0)

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

        overall_actions_only = wrapped.get_actions()
        overall_actions, intervals = wrapped2.get_actions()
        self.assertTrue(np.array_equal(overall_actions, overall_actions_only))
        self.assertTrue(overall_actions_only == [0, 1, 2, 3, 4])

        self.assertTrue(len(intervals) == 3)
        self.assertTrue(len(intervals[0]) == 2)
        self.assertTrue(intervals[0] == [0, 1])
        self.assertTrue(len(intervals[1]) == 2)
        self.assertTrue(intervals[1] == [2, 3])
        self.assertTrue(len(intervals[2]) == 1)
        self.assertTrue(intervals[2] == [4])

    def test_rendering(self):
        bench = FastDownwardBenchmark()
        env = bench.get_benchmark_env()
        wrapped = ActionFrequencyWrapper(env, 2)
        wrapped.reset()
        for _ in range(10):
            wrapped.step(1)
        img = wrapped.render_action_tracking()
        self.assertTrue(img.shape[-1] == 3)

        bench = CMAESBenchmark()
        env = bench.get_benchmark_env()
        wrapped = ActionFrequencyWrapper(env, 2)
        wrapped.reset()
        wrapped.step(np.ones(10))
        img = wrapped.render_action_tracking()
        self.assertTrue(img.shape[-1] == 3)

        class dict_action_env:
            def __init__(self):
                self.action_space = gym.spaces.Dict(
                    {
                        "one": gym.spaces.Discrete(2),
                        "two": gym.spaces.Box(
                            low=np.array([-1, 1]), high=np.array([1, 5])
                        ),
                    }
                )
                self.observation_space = gym.spaces.Discrete(2)
                self.reward_range = (1, 2)
                self.metadata = {}

            def reset(self):
                return 1

            def step(self, action):
                return 1, 1, 1, 1

        env = dict_action_env()
        wrapped = ActionFrequencyWrapper(env)
        wrapped.reset()
        with pytest.raises(NotImplementedError):
            wrapped.render_action_tracking()

        class tuple_action_env:
            def __init__(self):
                self.action_space = gym.spaces.Tuple(
                    (
                        gym.spaces.Discrete(2),
                        gym.spaces.Box(low=np.array([-1, 1]), high=np.array([1, 5])),
                    )
                )
                self.observation_space = gym.spaces.Discrete(2)
                self.reward_range = (1, 2)
                self.metadata = {}

            def reset(self):
                return 1

            def step(self, action):
                return 1, 1, 1, 1

        env = tuple_action_env()
        wrapped = ActionFrequencyWrapper(env)
        wrapped.reset()
        with pytest.raises(NotImplementedError):
            wrapped.render_action_tracking()

        class multi_discrete_action_env:
            def __init__(self):
                self.action_space = gym.spaces.MultiDiscrete([2, 3])
                self.observation_space = gym.spaces.Discrete(2)
                self.reward_range = (1, 2)
                self.metadata = {}

            def reset(self):
                return 1

            def step(self, action):
                return 1, 1, 1, 1

        env = multi_discrete_action_env()
        wrapped = ActionFrequencyWrapper(env, 5)
        wrapped.reset()
        for _ in range(10):
            wrapped.step([1, 2])
        img = wrapped.render_action_tracking()
        self.assertTrue(img.shape[-1] == 3)

        class multi_binary_action_env:
            def __init__(self):
                self.action_space = gym.spaces.MultiBinary(2)
                self.observation_space = gym.spaces.Discrete(2)
                self.reward_range = (1, 2)
                self.metadata = {}

            def reset(self):
                return 1

            def step(self, action):
                return 1, 1, 1, 1

        env = multi_binary_action_env()
        wrapped = ActionFrequencyWrapper(env)
        wrapped.reset()
        wrapped.step([1, 0])
        img = wrapped.render_action_tracking()
        self.assertTrue(img.shape[-1] == 3)
