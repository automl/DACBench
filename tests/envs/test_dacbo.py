from __future__ import annotations

import unittest

pytest = __import__("pytest")
dacboenv = pytest.importorskip("dacboenv")

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

    def test_close(self):
        env = self.make_env()
        env.reset()
        env.close()
