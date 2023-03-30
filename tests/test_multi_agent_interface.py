import unittest

import numpy as np

from dacbench.benchmarks import ModCMABenchmark, SigmoidBenchmark, ToySGDBenchmark
from dacbench.envs import SigmoidEnv, ToySGDEnv


class TestMultiAgentInterface(unittest.TestCase):
    def test_make_env(self):
        bench = SigmoidBenchmark()
        bench.config["multi_agent"] = True
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), SigmoidEnv))

        bench = ToySGDBenchmark()
        bench.config["multi_agent"] = True
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), ToySGDEnv))

    def test_empty_reset_step(self):
        bench = ModCMABenchmark()
        bench.config["multi_agent"] = True
        env = bench.get_environment()
        out = env.reset()
        self.assertTrue(out is None)
        env.register_agent(max(env.possible_agents))
        out = env.step(0)
        self.assertTrue(out is None)

    def test_last(self):
        bench = ModCMABenchmark()
        bench.config["multi_agent"] = True
        env = bench.get_environment()
        env.reset()
        state, reward, terminated, truncated, info = env.last()
        self.assertFalse(state is None)
        self.assertTrue(reward is None)
        self.assertFalse(info is None)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        env.register_agent(max(env.possible_agents))
        env.step(0)
        _, reward, _, _, info = env.last()
        self.assertFalse(reward is None)
        self.assertFalse(info is None)

    def test_agent_registration(self):
        bench = SigmoidBenchmark()
        bench.config["multi_agent"] = True
        env = bench.get_environment()
        env.reset()
        state, _, _, _, _ = env.last()
        env.register_agent(0)
        env.register_agent(max(env.possible_agents))
        self.assertTrue(len(env.agents) == 2)
        self.assertTrue(0 in env.agents)
        self.assertTrue(max(env.possible_agents) in env.agents)
        self.assertTrue(env.current_agent == 0)
        env.step(0)
        state2, _, _, _, _ = env.last()
        self.assertTrue(np.array_equal(state, state2))
        self.assertTrue(env.current_agent == max(env.possible_agents))
        env.step(1)
        state3, _, _, _, _ = env.last()
        self.assertFalse(np.array_equal(state, state3))
        env.remove_agent(0)
        self.assertTrue(len(env.agents) == 1)
        self.assertFalse(0 in env.agents)
        env.register_agent("value_dim_0")
        self.assertTrue(len(env.agents) == 2)
        self.assertTrue(0 in env.agents)
