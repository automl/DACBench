import unittest
from gym import spaces
import os
import numpy as np

# import shutil
from dacbench.runner import run_dacbench  # , plot_results
from dacbench.abstract_agent import AbstractDACBenchAgent
import matplotlib

matplotlib.use("Agg")


class TestRunner(unittest.TestCase):
    def test_abstract_agent(self):
        agent = AbstractDACBenchAgent("dummy")

        with self.assertRaises(NotImplementedError):
            agent.act(0, 0)

        with self.assertRaises(NotImplementedError):
            agent.train(0, 0)

        with self.assertRaises(NotImplementedError):
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

        run_dacbench("test_run", make, 1, ["LubyBenchmark", "SigmoidBenchmark"])
        self.assertTrue(os.path.exists("test_run"))
        self.assertTrue(os.path.exists("test_run/LubyBenchmark/seed_9"))
        self.assertTrue(os.path.exists("test_run/SigmoidBenchmark/seed_9"))


#    def test_plotting(self):
#        plot_results("test_run")
#        shutil.rmtree("test_run", ignore_errors=True)
