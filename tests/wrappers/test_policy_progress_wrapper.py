from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from dacbench.benchmarks import SigmoidBenchmark
from dacbench.wrappers import PolicyProgressWrapper


def _sig(x, scaling, inflection):
    return 1 / (1 + np.exp(-scaling * (x - inflection)))


def compute_optimal_sigmoid(instance):
    sig_values = [_sig(i, instance[1], instance[0]) for i in range(10)]
    optimal = [np.around(x) for x in sig_values]
    return [optimal]


class TestPolicyProgressWrapper(unittest.TestCase):
    def test_init(self):
        bench = SigmoidBenchmark()
        bench.set_action_values((3,))
        env = bench.get_environment()
        wrapped = PolicyProgressWrapper(env, compute_optimal_sigmoid)
        assert len(wrapped.policy_progress) == 0
        assert len(wrapped.episode) == 0
        assert wrapped.compute_optimal is not None

    def test_step(self):
        bench = SigmoidBenchmark()
        bench.set_action_values((3,))
        bench.config.instance_set = {0: [0, 0], 1: [1, 1], 2: [3, 4], 3: [5, 6]}
        env = bench.get_environment()
        wrapped = PolicyProgressWrapper(env, compute_optimal_sigmoid)

        wrapped.reset()
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = wrapped.step(action)
        assert len(wrapped.episode) == 1
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = wrapped.step(action)
        assert len(wrapped.episode) == 0
        assert len(wrapped.policy_progress) == 1

    @mock.patch("dacbench.wrappers.policy_progress_wrapper.plt")
    def test_render(self, mock_plt):
        bench = SigmoidBenchmark()
        bench.set_action_values((3,))
        env = bench.get_environment()
        env = PolicyProgressWrapper(env, compute_optimal_sigmoid)
        for _ in range(2):
            terminated, truncated = False, False
            env.reset()
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        env.render_policy_progress()
        assert mock_plt.show.called
