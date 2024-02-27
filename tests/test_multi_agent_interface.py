from __future__ import annotations

import unittest

import numpy as np
from dacbench.benchmarks import SigmoidBenchmark, ToySGDBenchmark
from dacbench.envs import SigmoidEnv, ToySGDEnv


class TestMultiAgentInterface(unittest.TestCase):
    def test_make_env(self):
        bench = SigmoidBenchmark()
        bench.config["multi_agent"] = True
        env = bench.get_environment()
        assert issubclass(type(env), SigmoidEnv)

        bench = ToySGDBenchmark()
        bench.config["multi_agent"] = True
        env = bench.get_environment()
        assert issubclass(type(env), ToySGDEnv)

    def test_empty_reset_step(self):
        bench = SigmoidBenchmark()
        bench.config["multi_agent"] = True
        env = bench.get_environment()
        out = env.reset()
        assert out is None
        env.register_agent(max(env.possible_agents))
        out = env.step(0)
        assert out is None

    def test_last(self):
        bench = ToySGDBenchmark()
        bench.config["multi_agent"] = True
        env = bench.get_environment()
        env.reset()
        state, reward, terminated, truncated, info = env.last()
        assert state is not None
        assert reward is None
        assert info is not None
        assert not terminated
        assert not truncated
        env.register_agent(max(env.possible_agents))
        env.step(0)
        _, reward, _, _, info = env.last()
        assert reward is not None
        assert info is not None

    def test_agent_registration(self):
        bench = SigmoidBenchmark()
        bench.config["multi_agent"] = True
        env = bench.get_environment()
        env.reset()
        state, _, _, _, _ = env.last()
        env.register_agent(0)
        env.register_agent(max(env.possible_agents))
        assert len(env.agents) == 2
        assert 0 in env.agents
        assert max(env.possible_agents) in env.agents
        assert env.current_agent == 0
        env.step(0)
        state2, _, _, _, _ = env.last()
        assert np.array_equal(state, state2)
        assert env.current_agent == max(env.possible_agents)
        env.step(1)
        state3, _, _, _, _ = env.last()
        assert not np.array_equal(state, state3)
        env.remove_agent(0)
        assert len(env.agents) == 1
        assert 0 not in env.agents
        env.register_agent("value_dim_0")
        assert len(env.agents) == 2
        assert 0 in env.agents
