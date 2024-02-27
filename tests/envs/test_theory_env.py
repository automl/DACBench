from __future__ import annotations

import unittest

import gymnasium as gym
from dacbench.benchmarks import TheoryBenchmark


class TestTheoryEnv(unittest.TestCase):
    def test_discrete_env(self):
        bench = TheoryBenchmark(
            config={
                "discrete_action": True,
                "action_choices": [1, 2, 4, 8],
                "instance_set_path": "lo_rls_50.csv",
            }
        )
        env = bench.get_environment()

        # check observation space
        s, _ = env.reset()  # default observation space: n, f(x)
        assert len(s) == 2
        assert s[0] == env.n
        assert s[1] == env.x.fitness

        # check action space
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 4

        # check instance-specific cutoff time
        assert env.max_evals == int(0.8 * env.n * env.n)

        # check initial solution for various instances
        for _i in range(len(env.instance_set)):
            if env.instance.initObj != "random":
                assert int(env.x.fitness) == int(env.instance.initObj)
            env.reset()

    def test_non_discrete_env(self):
        bench = TheoryBenchmark(
            config={
                "discrete_action": False,
                "min_action": 1,
                "max_action": 49,
                "instance_set_path": "lo_rls_50.csv",
            }
        )
        env = bench.get_environment()

        # check observation space
        s, _ = env.reset()  # default observation space: n, f(x)
        assert len(s) == 2
        assert s[0] == env.n
        assert s[1] == env.x.fitness

        # check action space
        assert isinstance(env.action_space, gym.spaces.Box)

        # check instance-specific cutoff time
        assert env.max_evals == int(0.8 * env.n * env.n)

        # check initial solution for various instances
        for _i in range(len(env.instance_set)):
            if env.instance.initObj != "random":
                assert int(env.x.fitness) == int(env.instance.initObj)
            env.reset()

        # check behaviour with out-of-range action
        s, r, terminated, truncated, info = env.step(
            100
        )  # a large negative reward will be returned and the epsidoe will end
        assert r < -1e4
        assert terminated or truncated


if __name__ == "__main__":
    TestTheoryEnv().test_discrete_env()
    TestTheoryEnv().test_non_discrete_env()
