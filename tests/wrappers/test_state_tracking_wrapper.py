import tempfile
import unittest
from itertools import groupby
from pathlib import Path

import gymnasium as gym
import numpy as np

from dacbench.agents import StaticAgent
from dacbench.benchmarks import CMAESBenchmark, LubyBenchmark
from dacbench.logger import Logger, load_logs, log2dataframe
from dacbench.runner import run_benchmark
from dacbench.wrappers import StateTrackingWrapper


class TestStateTrackingWrapper(unittest.TestCase):
    def test_box_logging(self):
        temp_dir = tempfile.TemporaryDirectory()

        seed = 0
        episodes = 10
        logger = Logger(
            output_path=Path(temp_dir.name),
            experiment_name="test_box_logging",
            step_write_frequency=None,
            episode_write_frequency=1,
        )

        bench = LubyBenchmark()
        bench.set_seed(seed)
        env = bench.get_environment()
        state_logger = logger.add_module(StateTrackingWrapper)
        wrapped = StateTrackingWrapper(env, logger=state_logger)
        agent = StaticAgent(env, 1)
        logger.set_env(env)

        run_benchmark(wrapped, agent, episodes, logger)
        state_logger.close()

        logs = load_logs(state_logger.get_logfile())
        dataframe = log2dataframe(logs, wide=True)

        sate_columns = [
            "state_Action t (current)",
            "state_Step t (current)",
            "state_Action t-1",
            "state_Action t-2",
            "state_Step t-1",
            "state_Step t-2",
        ]

        for state_column in sate_columns:
            self.assertTrue(state_column in dataframe.columns)
            self.assertTrue((~dataframe[state_column].isna()).all())

        temp_dir.cleanup()

    def test_dict_logging(self):
        temp_dir = tempfile.TemporaryDirectory()

        seed = 0
        episodes = 2
        logger = Logger(
            output_path=Path(temp_dir.name),
            experiment_name="test_dict_logging",
            step_write_frequency=None,
            episode_write_frequency=1,
        )

        bench = CMAESBenchmark()
        bench.set_seed(seed)
        env = bench.get_environment()
        state_logger = logger.add_module(StateTrackingWrapper)
        wrapped = StateTrackingWrapper(env, logger=state_logger)
        agent = StaticAgent(env, 3.5)
        logger.set_env(env)

        run_benchmark(wrapped, agent, episodes, logger)
        state_logger.close()

        logs = load_logs(state_logger.get_logfile())
        dataframe = log2dataframe(logs, wide=False)
        state_parts = {
            "Loc": 10,
            "Past Deltas": 40,
            "Population Size": 1,
            "Sigma": 1,
            "History Deltas": 80,
            "Past Sigma Deltas": 40,
        }

        names = dataframe.name.unique()

        def field(name: str):
            state, field_, *idx = name.split("_")
            return field_

        parts = groupby(sorted(names), key=field)

        for part, group_members in parts:
            expected_number = state_parts[part]
            actual_number = len(list(group_members))

            self.assertEqual(expected_number, actual_number)

        temp_dir.cleanup()

    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = StateTrackingWrapper(env)
        self.assertTrue(len(wrapped.overall_states) == 0)
        self.assertTrue(wrapped.state_interval is None)
        wrapped.instance = [0]
        self.assertTrue(wrapped.instance[0] == 0)

        wrapped2 = StateTrackingWrapper(env, 10)
        self.assertTrue(len(wrapped2.overall_states) == 0)
        self.assertTrue(wrapped2.state_interval == 10)
        self.assertTrue(len(wrapped2.state_intervals) == 0)
        self.assertTrue(len(wrapped2.current_states) == 0)

    def test_step_reset(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = StateTrackingWrapper(env, 2)

        state, info = wrapped.reset()
        self.assertTrue(issubclass(type(info), dict))
        self.assertTrue(len(state) > 1)
        self.assertTrue(len(wrapped.overall_states) == 1)

        state, reward, terminated, truncated, _ = wrapped.step(1)
        self.assertTrue(len(state) > 1)
        self.assertTrue(reward <= 0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        self.assertTrue(len(wrapped.overall_states) == 2)
        self.assertTrue(len(wrapped.current_states) == 2)
        self.assertTrue(len(wrapped.state_intervals) == 0)

        state, _ = wrapped.reset()
        self.assertTrue(len(wrapped.overall_states) == 3)
        self.assertTrue(len(wrapped.current_states) == 1)
        self.assertTrue(len(wrapped.state_intervals) == 1)

    def test_get_states(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = StateTrackingWrapper(env)
        wrapped.reset()
        for i in range(4):
            wrapped.step(i)
        wrapped2 = StateTrackingWrapper(env, 2)
        wrapped2.reset()
        for i in range(4):
            wrapped2.step(i)

        overall_states_only = wrapped.get_states()
        overall_states, intervals = wrapped2.get_states()
        self.assertTrue(np.array_equal(overall_states, overall_states_only))
        self.assertTrue(len(overall_states_only) == 5)
        self.assertTrue(len(overall_states_only[4]) == 6)

        self.assertTrue(len(intervals) == 3)
        self.assertTrue(len(intervals[0]) == 2)
        self.assertTrue(len(intervals[1]) == 2)
        self.assertTrue(len(intervals[2]) == 1)

    def test_rendering(self):
        bench = CMAESBenchmark()
        env = bench.get_environment()
        wrapped = StateTrackingWrapper(env)
        wrapped.reset()
        with self.assertRaises(NotImplementedError):
            wrapped.render_state_tracking()

        bench = CMAESBenchmark()

        def dummy(_):
            return [1, [2, 3]]

        bench.config.state_method = dummy
        bench.config.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(2),
                gym.spaces.Box(low=np.array([-1, 1]), high=np.array([5, 5])),
            )
        )
        env = bench.get_environment()
        wrapped = StateTrackingWrapper(env)
        wrapped.reset()
        with self.assertRaises(NotImplementedError):
            wrapped.render_state_tracking()

        def dummy2(_):
            return [0.5]

        bench.config.state_method = dummy2
        bench.config.observation_space = gym.spaces.Box(
            low=np.array([0]), high=np.array([1])
        )
        env = bench.get_environment()
        wrapped = StateTrackingWrapper(env)
        wrapped.reset()
        wrapped.step(1)
        wrapped.step(1)
        img = wrapped.render_state_tracking()
        self.assertTrue(img.shape[-1] == 3)

        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = StateTrackingWrapper(env, 2)
        wrapped.reset()
        wrapped.step(1)
        wrapped.step(1)
        img = wrapped.render_state_tracking()
        self.assertTrue(img.shape[-1] == 3)

        class discrete_obs_env:
            def __init__(self):
                self.observation_space = gym.spaces.Discrete(2)
                self.action_space = gym.spaces.Discrete(2)
                self.reward_range = (1, 2)
                self.metadata = {}

            def reset(self):
                return 1, {}

            def step(self, _):
                return 1, 1, 1, 1, {}

        env = discrete_obs_env()
        wrapped = StateTrackingWrapper(env, 2)
        wrapped.reset()
        wrapped.step(1)
        img = wrapped.render_state_tracking()
        self.assertTrue(img.shape[-1] == 3)

        class multi_discrete_obs_env:
            def __init__(self):
                self.observation_space = gym.spaces.MultiDiscrete([2, 3])
                self.action_space = gym.spaces.Discrete(2)
                self.reward_range = (1, 2)
                self.metadata = {}

            def reset(self):
                return [1, 2], {}

            def step(self, _):
                return [1, 2], 1, 1, 1, {}

        env = multi_discrete_obs_env()
        wrapped = StateTrackingWrapper(env)
        wrapped.reset()
        wrapped.step(1)
        img = wrapped.render_state_tracking()
        self.assertTrue(img.shape[-1] == 3)

        class multi_binary_obs_env:
            def __init__(self):
                self.observation_space = gym.spaces.MultiBinary(2)
                self.action_space = gym.spaces.Discrete(2)
                self.reward_range = (1, 2)
                self.metadata = {}

            def reset(self):
                return [1, 1], {}

            def step(self, _):
                return [1, 1], 1, 1, 1, {}

        env = multi_binary_obs_env()
        wrapped = StateTrackingWrapper(env)
        wrapped.reset()
        wrapped.step(1)
        img = wrapped.render_state_tracking()
        self.assertTrue(img.shape[-1] == 3)
