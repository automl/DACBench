import pytest
import unittest
from gym import spaces
import os
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
                    self.num_actions = env.action_space.n
                elif isinstance(env.action_space, spaces.MultiDiscrete):
                    self.num_actions = env.action_space.nvec
                else:
                    self.num_actions = int(env.action_space.high[0])

            def act(self, reward, state):
                return 1

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
