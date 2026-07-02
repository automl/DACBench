from __future__ import annotations

import logging
import tempfile
import unittest

import numpy as np

pytest = __import__("pytest")
pytest.importorskip("smac")

from gymnasium.spaces import Box, Dict, Discrete

from dacbench import AbstractEnv
from dacbench.benchmarks import DACBOBenchmark


class TestDACBOEnv(unittest.TestCase):
    def make_env(self, **overrides):
        bench = DACBOBenchmark()
        bench.config["instance_set"] = {0: (1, "bbob/2/1/0")}
        bench.config["task_ids"] = ["bbob/2/1/0"]
        bench.config["inner_seeds"] = [1]
        bench.config["evaluation_mode"] = True
        for k, v in overrides.items():
            bench.config[k] = v
        return bench.get_environment()

    def test_setup(self):
        env = self.make_env()
        assert issubclass(type(env), AbstractEnv)
        assert env._env is not None
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")
        assert len(env.instance_set) > 0

    def test_reset(self):
        env = self.make_env()
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert isinstance(env.instance, tuple)
        assert len(env.instance) == 2
        assert isinstance(env.observation_space, Dict)
        for key in obs:
            assert key in env.observation_space.spaces

    def test_step(self):
        env = self.make_env()
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_instance_selection(self):
        bench = DACBOBenchmark()
        bench.config["instance_set"] = {
            0: (1, "bbob/2/1/0"),
            1: (1, "bbob/2/2/0"),
            2: (1, "bbob/2/3/0"),
        }
        bench.config["task_ids"] = ["bbob/2/1/0", "bbob/2/2/0", "bbob/2/3/0"]
        bench.config["inner_seeds"] = [1]
        bench.config["evaluation_mode"] = True
        env = bench.get_environment()

        instances = list(bench.config["instance_set"].values())
        n = len(instances)
        # AbstractEnv.__init__ sets instance_index=0, each reset() advances
        # by 1 (round_robin), so first reset gives instance at index 1.
        # Loop 2 full cycles to verify wrap-around.
        for i in range(2 * n):
            env.reset()
            expected = instances[(i + 1) % n]
            assert env.instance == expected, (
                f"Episode {i}: DACBench instance "
                f"{env.instance} != expected {expected}"
            )
            assert env._env.instance == env.instance, (
                f"Episode {i}: inner env instance "
                f"{env._env.instance} != outer {env.instance}"
            )

    def test_external_instance_selector(self):
        from dacbench.envs.dacboenv.env.instance import ExternalInstanceSelector

        selector = ExternalInstanceSelector(
            task_ids=["task_a", "task_b"],
            seeds=[0, 1],
        )
        # Without set_instance, returns instances[0]
        first = selector.select_instance()
        assert selector.idx == 0
        second = selector.select_instance()
        assert selector.idx == 0
        assert first == second

        # After set_instance, returns the externally set instance
        custom = (42, "custom_task")
        selector.set_instance(custom)
        assert selector.select_instance() == custom
        assert selector.idx == 0  # idx unchanged

    def test_close(self):
        env = self.make_env()
        env.reset()
        env.close()

    def test_full_episode(self):
        env = self.make_env()
        env.reset()
        truncated = False
        info = {}
        steps = 0
        while not truncated and steps < 500:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        assert truncated, f"Episode did not truncate within {steps} steps"
        assert "episode" in info
        assert "r" in info["episode"]
        assert "l" in info["episode"]

    def test_action_space_bucket(self):
        from dacbench.envs.dacboenv.env.action import AcqParameterActionSpace

        env = self.make_env(
            action_space_class=AcqParameterActionSpace,
            action_space_kwargs={"adjustment_type": "bucket", "bounds": (-5, 5)},
            observation_keys=["trials_passed", "trials_left"],
        )
        env.reset()
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 11
        env.step(0)
        env.step(10)

    def test_action_space_step(self):
        from dacbench.envs.dacboenv.env.action import AcqParameterActionSpace

        env = self.make_env(
            action_space_class=AcqParameterActionSpace,
            action_space_kwargs={"adjustment_type": "step", "bounds": (-10, 10)},
            observation_keys=["trials_passed", "trials_left"],
        )
        env.reset()
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 3
        for a in [0, 1, 2]:
            env.step(a)

    def test_action_space_continuousstep(self):
        from dacbench.envs.dacboenv.env.action import AcqParameterActionSpace

        env = self.make_env(
            action_space_class=AcqParameterActionSpace,
            action_space_kwargs={
                "adjustment_type": "continuousstep",
                "bounds": (-5.0, 5.0),
            },
        )
        env.reset()
        assert isinstance(env.action_space, Box)
        assert env._env._action_space._last == 0.0
        env.step(np.array([1.0], dtype=np.float32))
        assert env._env._action_space._last == pytest.approx(1.0)

    def test_acq_function_action_space(self):
        from dacbench.envs.dacboenv.env.action import AcqFunctionActionSpace

        env = self.make_env(
            action_space_class=AcqFunctionActionSpace,
            action_space_kwargs={},
            observation_keys=["trials_passed", "trials_left"],
        )
        env.reset()
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 3
        for a in range(3):
            env.step(a)

    def test_wei_temporld_invalid_step_durations(self):
        from dacbench.envs.dacboenv.env.action import WEITempoRLActionSpace

        with pytest.raises(ValueError, match="step_durations must be > 0"):
            WEITempoRLActionSpace(smac_instance=None, step_durations=[0, 1, 5])

    def test_observation_keys_subset(self):
        env = self.make_env(observation_keys=["trials_passed", "trials_left"])
        obs, _ = env.reset()
        assert set(obs.keys()) == {"trials_passed", "trials_left"}


class TestInstanceSelectors(unittest.TestCase):
    def test_round_robin_instance_selector(self):
        from dacbench.envs.dacboenv.env.instance import RoundRobinInstanceSelector

        task_ids = ["a", "b", "c"]
        seeds = [0]
        selector = RoundRobinInstanceSelector(task_ids=task_ids, seeds=seeds)
        instances = selector.instances  # [(0, "a"), (0, "b"), (0, "c")]

        for i in range(6):
            inst = selector.select_instance()
            expected = instances[i % len(instances)]
            assert inst == expected, f"Step {i}: got {inst}, expected {expected}"

    def test_random_instance_selector(self):
        from dacbench.envs.dacboenv.env.instance import RandomInstanceSelector

        task_ids = ["a", "b", "c", "d", "e"]
        seeds = [0, 1]

        # Same seed gives same sequence
        sel1 = RandomInstanceSelector(task_ids=task_ids, seeds=seeds, selector_seed=42)
        sel2 = RandomInstanceSelector(task_ids=task_ids, seeds=seeds, selector_seed=42)
        assert sel1.select_instance() == sel2.select_instance()

        # Different seeds give different sequences
        sel3 = RandomInstanceSelector(task_ids=task_ids, seeds=seeds, selector_seed=0)
        sel4 = RandomInstanceSelector(task_ids=task_ids, seeds=seeds, selector_seed=1)
        results3 = [sel3.select_instance() for _ in range(5)]
        results4 = [sel4.select_instance() for _ in range(5)]
        assert results3 != results4


class TestFeatures(unittest.TestCase):
    def test_knn_entropy_basic(self):
        from dacbench.envs.dacboenv.features.X_features import knn_entropy

        X = np.random.default_rng(42).random((20, 2))
        result = knn_entropy(X)
        assert isinstance(float(result), float)

    def test_exploration_tsd_basic(self):
        from dacbench.envs.dacboenv.features.X_features import exploration_tsd

        X = np.random.default_rng(42).random((10, 2))
        result = exploration_tsd(X)
        assert result.shape == (10,)
        assert np.all(result[1:] >= result[:-1])

    def test_calculate_ubr_unfitted_model(self):
        from pathlib import Path

        from ConfigSpace import ConfigurationSpace
        from ConfigSpace.hyperparameters import UniformFloatHyperparameter
        from smac import BlackBoxFacade, Scenario

        from dacbench.envs.dacboenv.features.signal.ubr import calculate_ubr

        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformFloatHyperparameter("x", 0.0, 1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            scenario = Scenario(
                configspace=cs,
                n_trials=10,
                seed=42,
                output_directory=Path(tmpdir),
            )
            facade = BlackBoxFacade(
                scenario=scenario,
                target_function=lambda config, seed=0: 0.0,
                overwrite=True,
                logging_level=logging.WARNING,
            )
            smbo = facade.optimizer
            result = calculate_ubr(None, None, None, smbo=smbo)

        assert result == {}
