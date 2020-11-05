import pytest
import unittest
from gym import spaces
import os
import numpy as np
import shutil
from dacbench.runner import AbstractDACBenchAgent, run_dacbench, plot_results
import matplotlib

matplotlib.use("Agg")


class TestRunner(unittest.TestCase):
    def test_abstract_agen(self):
        agent = AbstractDACBenchAgent("dummy")

        with pytest.raises(NotImplementedError):
            agent.act(0, 0)

        with pytest.raises(NotImplementedError):
            agent.train(0, 0)

        with pytest.raises(NotImplementedError):
            agent.end_episode(0, 0)

    def test_loop(self):
        class DummyAgent(AbstractDACBenchAgent):
            def __init__(self, env):
                if isinstance(env.action_space, spaces.Discrete):
                    self.num_actions = 1
                elif isinstance(env.action_space, spaces.MultiDiscrete):
                    self.num_actions = len(env.action_space.nvec)
                else:
                    self.num_actions = len(env.action_space.high)

            def act(self, reward, state):
                action = np.ones(self.num_actions)
                print(self.num_actions)
                if self.num_actions == 1:
                    action = 1
                return action

            def train(self, reward, state):
                pass

            def end_episode(self, reward, state):
                pass

        def make(env):
            return DummyAgent(env)

        run_dacbench("test_run", make, 1)
        self.assertTrue(os.path.exists("test_run"))
        self.assertFalse(os.stat("test_run/LubyBenchmark.json") == 0)
        self.assertFalse(os.stat("test_run/SigmoidBenchmark.json") == 0)
        self.assertFalse(os.stat("test_run/CMAESBenchmark.json") == 0)
        self.assertFalse(os.stat("test_run/FastDownwardBenchmark.json") == 0)

    def test_plotting(self):
        plot_results("test_run")
        shutil.rmtree("test_run", ignore_errors=True)
