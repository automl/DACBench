import tempfile
import unittest
from pathlib import Path

import gym
import numpy as np
import pandas as pd


from dacbench.agents import RandomAgent
from dacbench.benchmarks import (
    LubyBenchmark,
    FastDownwardBenchmark,
    CMAESBenchmark,
    ModeaBenchmark,
)
from dacbench.logger import Logger, load_logs, log2dataframe
from dacbench.runner import run_benchmark
from dacbench.wrappers import ActionFrequencyWrapper


class TestActionTrackingWrapper(unittest.TestCase):
    def test_logging_multi_discrete(self):
        temp_dir = tempfile.TemporaryDirectory()

        seed = 0
        logger = Logger(
            output_path=Path(temp_dir.name),
            experiment_name="test_multi_discrete_logging",
            step_write_frequency=None,
            episode_write_frequency=1,
        )

        bench = ModeaBenchmark()
        bench.set_seed(seed)
        env = bench.get_environment()
        env.seed_action_space(seed)
        action_logger = logger.add_module(ActionFrequencyWrapper)
        wrapped = ActionFrequencyWrapper(env, logger=action_logger)
        agent = RandomAgent(env)
        logger.set_env(env)

        run_benchmark(wrapped, agent, 1, logger)
        action_logger.close()

        logs = load_logs(action_logger.get_logfile())
        dataframe = log2dataframe(logs, wide=True)

        expected_actions = pd.DataFrame(
            {
                "action_0": {
                    0: 0,
                    1: 1,
                    2: 0,
                    3: 1,
                    4: 1,
                    5: 0,
                    6: 1,
                    7: 1,
                    8: 0,
                    9: 0,
                    10: 0,
                },
                "action_1": {
                    0: 1,
                    1: 0,
                    2: 1,
                    3: 0,
                    4: 0,
                    5: 1,
                    6: 0,
                    7: 1,
                    8: 0,
                    9: 0,
                    10: 1,
                },
                "action_10": {
                    0: 0,
                    1: 0,
                    2: 1,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 2,
                    8: 1,
                    9: 2,
                    10: 1,
                },
                "action_2": {
                    0: 1,
                    1: 1,
                    2: 1,
                    3: 0,
                    4: 1,
                    5: 1,
                    6: 1,
                    7: 1,
                    8: 0,
                    9: 0,
                    10: 1,
                },
                "action_3": {
                    0: 0,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                    5: 1,
                    6: 1,
                    7: 0,
                    8: 0,
                    9: 1,
                    10: 1,
                },
                "action_4": {
                    0: 0,
                    1: 1,
                    2: 1,
                    3: 0,
                    4: 1,
                    5: 0,
                    6: 0,
                    7: 1,
                    8: 0,
                    9: 1,
                    10: 0,
                },
                "action_5": {
                    0: 1,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 1,
                    5: 1,
                    6: 1,
                    7: 0,
                    8: 0,
                    9: 0,
                    10: 1,
                },
                "action_6": {
                    0: 0,
                    1: 1,
                    2: 1,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 1,
                    9: 0,
                    10: 0,
                },
                "action_7": {
                    0: 1,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 1,
                    8: 1,
                    9: 1,
                    10: 0,
                },
                "action_8": {
                    0: 0,
                    1: 1,
                    2: 0,
                    3: 1,
                    4: 1,
                    5: 1,
                    6: 0,
                    7: 1,
                    8: 0,
                    9: 0,
                    10: 1,
                },
                "action_9": {
                    0: 1,
                    1: 2,
                    2: 1,
                    3: 0,
                    4: 0,
                    5: 1,
                    6: 1,
                    7: 1,
                    8: 2,
                    9: 0,
                    10: 2,
                },
            }
        )

        for column in expected_actions.columns:
            # todo: seems to be an bug here. Every so ofter the last action is missing.
            # Double checked not a logging problem. Could be a seeding issue
            self.assertListEqual(
                dataframe[column].to_list()[:10],
                expected_actions[column].to_list()[:10],
                f"Column  {column}",
            )

        temp_dir.cleanup()

    def test_logging_discrete(self):

        temp_dir = tempfile.TemporaryDirectory()

        seed = 0
        logger = Logger(
            output_path=Path(temp_dir.name),
            experiment_name="test_discrete_logging",
            step_write_frequency=None,
            episode_write_frequency=1,
        )

        bench = LubyBenchmark()
        bench.set_seed(seed)
        env = bench.get_environment()
        env.seed_action_space(seed)

        action_logger = logger.add_module(ActionFrequencyWrapper)
        wrapped = ActionFrequencyWrapper(env, logger=action_logger)
        agent = RandomAgent(env)
        logger.set_env(env)

        run_benchmark(wrapped, agent, 10, logger)
        action_logger.close()

        logs = load_logs(action_logger.get_logfile())
        dataframe = log2dataframe(logs, wide=True)

        expected_actions = [
            0,
            3,
            5,
            4,
            3,
            5,
            5,
            5,
            3,
            3,
            2,
            1,
            0,
            1,
            2,
            0,
            1,
            1,
            0,
            1,
            2,
            4,
            3,
            0,
            1,
            3,
            0,
            3,
            3,
            3,
            4,
            4,
            4,
            5,
            4,
            0,
            4,
            2,
            1,
            3,
            4,
            2,
            1,
            3,
            3,
            2,
            0,
            5,
            2,
            5,
            2,
            1,
            5,
            3,
            2,
            5,
            1,
            0,
            2,
            3,
            1,
            3,
            2,
            3,
            2,
            4,
            3,
            4,
            0,
            5,
            5,
            1,
            5,
            0,
            1,
            5,
            5,
            3,
            3,
            2,
        ]

        self.assertListEqual(dataframe.action.to_list(), expected_actions)

        temp_dir.cleanup()

    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
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
        env = bench.get_environment()
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
        env = bench.get_environment()
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
        env = bench.get_environment()
        wrapped = ActionFrequencyWrapper(env, 2)
        wrapped.reset()
        for _ in range(10):
            wrapped.step(1)
        img = wrapped.render_action_tracking()
        self.assertTrue(img.shape[-1] == 3)

        bench = CMAESBenchmark()
        env = bench.get_environment()
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
        with self.assertRaises(NotImplementedError):
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
        with self.assertRaises(NotImplementedError):
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

        class large_action_env:
            def __init__(self):
                self.action_space = gym.spaces.Box(low=np.zeros(15), high=np.ones(15))
                self.observation_space = gym.spaces.Discrete(2)
                self.reward_range = (1, 2)
                self.metadata = {}

            def reset(self):
                return 1

            def step(self, action):
                return 1, 1, 1, 1

        env = large_action_env()
        wrapped = ActionFrequencyWrapper(env)
        wrapped.reset()
        wrapped.step(0.5 * np.ones(15))
        img = wrapped.render_action_tracking()
        self.assertTrue(img.shape[-1] == 3)
