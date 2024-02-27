from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
from dacbench.abstract_agent import AbstractDACBenchAgent
from dacbench.runner import run_dacbench
from gymnasium import spaces

matplotlib.use("Agg")


class TestRunner(unittest.TestCase):
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
            assert os.stat(path / "LubyBenchmark") != 0
            assert os.stat(path / "SigmoidBenchmark") != 0
