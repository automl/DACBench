"""Tests for the GA benchmark."""

from __future__ import annotations

import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
from dacbench.benchmarks import GABenchmark
from dacbench.benchmarks.ga_benchmark import GA_DEFAULTS
from dacbench.envs import GAEnv

GA_INSTANCE_DIR = (
    Path(__file__).resolve().parents[2] / "dacbench" / "instance_sets" / "ga"
)


class TestGABenchmark(unittest.TestCase):
    """Tests for GABenchmark."""

    def test_defaults(self):
        """Defaults instantiate a discrete one_plus_one OneMax env."""
        bench = GABenchmark()
        for key, val in GA_DEFAULTS.items():
            assert bench.config[key] == val
        env = bench.get_environment()
        assert isinstance(env, GAEnv)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 5
        assert env.observation_space.shape == (2,)
        env.reset()
        assert env.max_evals == int(0.8 * env.n * env.n)

    def test_read_instance_sets(self):
        """Every shipped instance set can be read."""
        # the directory also holds the derived policy tables (optimal_policy_*.csv)
        files = sorted(
            path
            for path in GA_INSTANCE_DIR.glob("*.csv")
            if not path.name.startswith("optimal_policy_")
        )
        assert len(files) == 17
        for path in files:
            bench = GABenchmark(config={"instance_set_path": path.name})
            instances = bench.config.instance_set
            assert len(instances) > 0
            for inst in instances.values():
                assert "size" in inst
                assert "initObj" in inst

    def test_read_instance_set_from_cwd_path_and_test_set(self):
        """Paths are resolved as given first; test_set_path is loaded too."""
        path = str(GA_INSTANCE_DIR / "onemax_100.csv")
        bench = GABenchmark(
            config={"instance_set_path": path, "test_set_path": "onemax_20.csv"}
        )
        assert bench.config.instance_set[0].size == 100
        assert bench.config.test_set[0].size == 20

    def test_test_env_cutoff(self):
        """A test env uses the configured cutoff as evaluation budget."""
        bench = GABenchmark(config={"cutoff": 1234})
        env = bench.get_environment(test_env=True)
        env.reset()
        assert env.max_evals == 1234

    def test_config_overrides(self):
        """action_choices and observation_description overrides are honoured."""
        bench = GABenchmark(
            config={
                "algorithm": "one_plus_lambda_lambda",
                "action_choices": [1, 2, 4, 8, 16, 32],
                "observation_description": "n, f(x), lambda, f(x)_{t-1}",
            }
        )
        env = bench.get_environment()
        assert env.action_space.n == 6
        assert env.observation_space.shape == (4,)
        s, _ = env.reset()
        assert len(s) == 4
        env.step(5)
        assert env.history_r[-1] == 32

    def test_continuous_one_plus_one(self):
        """Continuous r for one_plus_one defaults to [1, max size]."""
        bench = GABenchmark(config={"discrete_action": False})
        env = bench.get_environment()
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.low[0] == 1
        assert env.action_space.high[0] == 50
        env.reset()
        _, r, terminated, _, _ = env.step(np.array([2.7]))
        assert r > -1e4
        assert not terminated
        assert env.history_r[-1] == 2

    def test_gym_make(self):
        """The auto-registered GA-v0 env can be reset and stepped."""
        env = gym.make("GA-v0")
        s, _ = env.reset(seed=0)
        assert len(s) == 2
        s, _, terminated, truncated, _ = env.step(0)
        assert len(s) == 2
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
