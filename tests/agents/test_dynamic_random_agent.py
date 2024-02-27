from __future__ import annotations

import unittest

import numpy as np
from dacbench.agents import DynamicRandomAgent
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.wrappers import MultiDiscreteActionWrapper


class MyTestCase(unittest.TestCase):
    def get_agent(self, switching_interval):
        env = SigmoidBenchmark().get_benchmark()
        env = MultiDiscreteActionWrapper(env)
        env.action_space.seed(0)
        agent = DynamicRandomAgent(env, switching_interval=switching_interval)
        return agent, env

    def test_init(self):
        agent, _ = self.get_agent(switching_interval=4)
        assert agent.switching_interval == 4

    def test_deterministic(self):
        switching_interval = 2
        agent, env = self.get_agent(switching_interval)

        state, _ = env.reset()
        reward = 0
        actions = []
        for _ in range(6):
            action = agent.act(state, reward)
            state, reward, *_ = env.step(action)
            actions.append(action)

        agent, env = self.get_agent(switching_interval)
        state, _ = env.reset()
        reward = 0
        actions2 = []
        for _ in range(6):
            action = agent.act(state, reward)
            state, reward, *_ = env.step(action)
            actions2.append(action)

        assert actions == actions2

    def test_switing_interval(self):
        switching_interval = 3
        agent, env = self.get_agent(switching_interval)

        state, _ = env.reset()
        reward = 0
        actions = []
        for _i in range(21):
            action = agent.act(state, reward)
            state, reward, *_ = env.step(action)
            actions.append(action)

        actions = np.array(actions).reshape((-1, switching_interval))
        assert (actions[:, 0] == actions[:, 1]).all()
