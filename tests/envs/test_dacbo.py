from __future__ import annotations

import unittest

pytest = __import__("pytest")
pytest.importorskip("smac")

from gymnasium.spaces import Dict

from dacbench import AbstractEnv
from dacbench.benchmarks import DACBOBenchmark


class TestDACBOEnv(unittest.TestCase):
    def make_env(self):
        bench = DACBOBenchmark()
        bench.config["instance_set"] = {0: (1, "bbob/2/1/0")}
        bench.config["task_ids"] = ["bbob/2/1/0"]
        bench.config["inner_seeds"] = [1]
        bench.config["evaluation_mode"] = True
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
