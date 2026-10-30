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

    def test_cutoff_truncates(self):
        """Reaching the step cutoff truncates the episode."""
        bench = TheoryBenchmark()
        bench.config.cutoff = 3
        env = bench.get_environment()
        env.reset(seed=0)
        # Start from the all-zeros string so three steps cannot reach the optimum.
        env.x.data[:] = False
        env.x.eval()

        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())

        assert not env.x.is_optimal()
        assert truncated
        assert not terminated

    def test_optimum_terminates(self):
        """Reaching the optimum terminates the episode."""
        bench = TheoryBenchmark()
        env = bench.get_environment()
        env.reset(seed=0)
        # LeadingOnes optimum: every bit set.
        env.x.data[:] = True
        env.x.eval()

        _, _, terminated, truncated, _ = env.step(env.action_space.sample())

        assert env.x.is_optimal()
        assert terminated
        assert not truncated


if __name__ == "__main__":
    TestTheoryEnv().test_discrete_env()
    TestTheoryEnv().test_non_discrete_env()
