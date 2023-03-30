import os
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from gymnasium import spaces

from dacbench.abstract_agent import AbstractDACBenchAgent

# import shutil
from dacbench.runner import run_dacbench  # , plot_results

matplotlib.use("Agg")


class TestRunner(unittest.TestCase):
    def test_abstract_agent(self):
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
                if self.num_actions == 1:
                    action = 1
                return action

            def train(self, reward, state):
                pass

            def end_episode(self, reward, state):
                pass

        def make(env):
            return DummyAgent(env)

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dacbench(
                tmp_dir,
                make,
                1,
                bench=["LubyBenchmark", "SigmoidBenchmark"],
                seeds=[42],
            )
            path = Path(tmp_dir)
            self.assertFalse(os.stat(path / "LubyBenchmark") == 0)
            self.assertFalse(os.stat(path / "SigmoidBenchmark") == 0)


#    def test_plotting(self):
#        plot_results("test_run")
#        shutil.rmtree("test_run", ignore_errors=True)
