import unittest

from dacbench.agents import DynamicRandomAgent
from dacbench.benchmarks import SigmoidBenchmark
import numpy as np


class MyTestCase(unittest.TestCase):
    def get_agent(self, switching_interval):
        env = SigmoidBenchmark().get_benchmark()
        env.seed_action_space()

        agent = DynamicRandomAgent(env, switching_interval=switching_interval)
        return agent, env

    def test_init(self):
        agent, _ = self.get_agent(switching_interval=4)
        assert agent.switching_interval == 4

    def test_deterministic(self):
        switching_interval = 2
        agent, env = self.get_agent(switching_interval)

        state = env.reset()
        reward = 0
        actions = []
        for i in range(6):
            action = agent.act(state, reward)
            state, reward, *_ = env.step(action)
            actions.append(action)

        assert actions == [48, 48, 14, 14, 6, 6]

    def test_switing_interval(self):
        switching_interval = 3
        agent, env = self.get_agent(switching_interval)

        state = env.reset()
        reward = 0
        actions = []
        for i in range(21):
            action = agent.act(state, reward)
            state, reward, *_ = env.step(action)
            actions.append(action)

        actions = np.array(actions).reshape((-1, switching_interval))
        assert (actions[:, 0] == actions[:, 1]).all()
