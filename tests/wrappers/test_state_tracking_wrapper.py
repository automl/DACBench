from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from dacbench.agents import StaticAgent
from dacbench.benchmarks import LubyBenchmark
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
            assert state_column in dataframe.columns
            assert (~dataframe[state_column].isna()).all()

        temp_dir.cleanup()

    def test_init(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = StateTrackingWrapper(env)
        assert len(wrapped.overall_states) == 0
        assert wrapped.state_interval is None
        wrapped.instance = [0]
        assert wrapped.instance[0] == 0

        wrapped2 = StateTrackingWrapper(env, 10)
        assert len(wrapped2.overall_states) == 0
        assert wrapped2.state_interval == 10
        assert len(wrapped2.state_intervals) == 0
        assert len(wrapped2.current_states) == 0

    def test_step_reset(self):
        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = StateTrackingWrapper(env, 2)

        state, info = wrapped.reset()
        assert issubclass(type(info), dict)
        assert len(state) > 1
        assert len(wrapped.overall_states) == 1

        state, reward, terminated, truncated, _ = wrapped.step(1)
        assert len(state) > 1
        assert reward <= 0
        assert not terminated
        assert not truncated

        assert len(wrapped.overall_states) == 2
        assert len(wrapped.current_states) == 2
        assert len(wrapped.state_intervals) == 0

        state, _ = wrapped.reset()
        assert len(wrapped.overall_states) == 3
        assert len(wrapped.current_states) == 1
        assert len(wrapped.state_intervals) == 1

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
        assert np.array_equal(overall_states, overall_states_only)
        assert len(overall_states_only) == 5
        assert len(overall_states_only[4]) == 6

        assert len(intervals) == 3
        assert len(intervals[0]) == 2
        assert len(intervals[1]) == 2
        assert len(intervals[2]) == 1

    def test_rendering(self):
        bench = LubyBenchmark()

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
        with pytest.raises(NotImplementedError):
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
        assert img.shape[-1] == 3

        bench = LubyBenchmark()
        env = bench.get_environment()
        wrapped = StateTrackingWrapper(env, 2)
        wrapped.reset()
        wrapped.step(1)
        wrapped.step(1)
        img = wrapped.render_state_tracking()
        assert img.shape[-1] == 3

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
        assert img.shape[-1] == 3

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
        assert img.shape[-1] == 3

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
        assert img.shape[-1] == 3
