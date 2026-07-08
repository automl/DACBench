from __future__ import annotations

import json
import os
import unittest

import gymnasium as gym
from dacbench.abstract_benchmark import objdict
from dacbench.benchmarks import FunctionApproximationBenchmark
from dacbench.benchmarks.function_approximation_benchmark import (
    FUNCTION_APPROXIMATION_DEFAULTS,
    INFO,
)
from dacbench.envs import FunctionApproximationEnv, FunctionApproximationInstance


class TestFunctionApproximationBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = FunctionApproximationBenchmark()
        env = bench.get_environment()
        assert issubclass(type(env), FunctionApproximationEnv)

    def test_save_conf(self):
        bench = FunctionApproximationBenchmark()
        del bench.config["config_space"]
        bench.save_config("test_conf.json")
        with open("test_conf.json") as fp:
            recovered = json.load(fp)
        for k in bench.config:
            assert k in recovered
        os.remove("test_conf.json")

    def test_from_to_json(self):
        bench = FunctionApproximationBenchmark()
        restored_bench = FunctionApproximationBenchmark.from_json(bench.to_json())
        assert bench == restored_bench

    def test_read_instances(self):
        bench = FunctionApproximationBenchmark()
        bench.read_instance_set()
        assert len(bench.config.instance_set.keys()) == 300
        assert isinstance(bench.config.instance_set[0], FunctionApproximationInstance)
        first_inst = bench.config.instance_set[0]

        bench2 = FunctionApproximationBenchmark()
        env = bench2.get_environment()
        assert isinstance(bench.config.instance_set[0], FunctionApproximationInstance)
        assert env.instance_set[0].functions[0].a == first_inst.functions[0].a    
        assert env.instance_set[0].functions[0].b == first_inst.functions[0].b    
        assert len(env.instance_set.keys()) == 300

    def test_action_space_matches_discrete_bins(self):
        """Regression: action_space size must equal the per-dim bin count
        declared in `config.discrete` for every `get_benchmark(dimension=...)`.
        """
        bench = FunctionApproximationBenchmark()
        for dim in (1, 2, 3, 5):
            with self.subTest(dimension=dim):
                env = bench.get_benchmark(dimension=dim, seed=0)
                env.reset(seed=0)
                discrete = env.config.discrete
                action_space = env.action_space

                # The action space shape must mirror the bin count per dim.
                if isinstance(action_space, gym.spaces.Discrete):
                    # dimension == 1: a single Integer HP collapses to Discrete(n)
                    assert action_space.n == discrete[0]
                elif isinstance(action_space, gym.spaces.MultiDiscrete):
                    assert tuple(action_space.nvec) == tuple(discrete)
                else:  # Dict fallback for mixed-type config_spaces
                    names = list(action_space.spaces.keys())
                    assert len(names) == len(discrete)
                    for name, n_bins in zip(names, discrete, strict=True):
                        assert action_space.spaces[name].n == n_bins

                # Stepping with the largest valid index per dim must not raise
                # and must produce a state that fits the observation space.
                max_action = {
                    f"value_dim_{i + 1}": int(n_bins) - 1
                    for i, n_bins in enumerate(discrete)
                }
                state, _reward, _term, _trunc, _info = env.step(max_action)
                assert state.shape == env.observation_space.shape

    def test_observation_space_matches_state_description(self):
        """Regression: when observation_space_args is auto-derived from
        config_space, its length must match the actual state shape.

        Previously the formula `1 + len(config_space) * 4` hardcoded the
        assumption that every function exposes 3 instance_description entries
        and `omit_instance_type=False`. With `omit_instance_type=True` the
        identifier is dropped, so state shape is `1 + N*2 + N = 1 + 3N` while
        the old formula gives `1 + 4N`. The two diverge for any N>0 — this
        test uses the default config_space (N=2) and a state_description
        sized for `omit_instance_type=True` so the new formula
        (`len(state_description)`) gives the correct 7-element obs.
        """
        # 1 budget + 2 dims * 2 instance params + 2 actions = 7
        benchmark_info = objdict(FUNCTION_APPROXIMATION_DEFAULTS.benchmark_info)
        benchmark_info["state_description"] = [
            "Remaining Budget",
            "Function Parameter 1 (dimension 1)",
            "Function Parameter 2 (dimension 1)",
            "Function Parameter 1 (dimension 2)",
            "Function Parameter 2 (dimension 2)",
            "Action 1",
            "Action 2",
        ]
        bench = FunctionApproximationBenchmark(
            config=objdict(
                {
                    "config_space": FUNCTION_APPROXIMATION_DEFAULTS.config_space,
                    "omit_instance_type": True,
                    "benchmark_info": benchmark_info,
                }
            )
        )
        env = bench.get_environment()

        state, _info = env.reset(seed=0)
        assert state.shape == env.observation_space.shape
        assert env.observation_space.shape[0] == len(
            env.config.benchmark_info["state_description"]
        )

    def test_info_not_mutated_by_get_benchmark(self):
        """Regression: get_benchmark must not mutate the module-level INFO
        dict via shallow-copied benchmark_info. Previously, every call to
        `get_benchmark(dimension=...)` reassigned `self.config` to a shallow
        copy of FUNCTION_APPROXIMATION_DEFAULTS and then mutated
        `benchmark_info["state_description"]` — leaking into INFO and
        polluting subsequent benchmark constructions.
        """
        original = list(INFO["state_description"])
        try:
            bench = FunctionApproximationBenchmark()
            for dim in (1, 2, 3, 5):
                bench.get_benchmark(dimension=dim, seed=0)
            assert INFO["state_description"] == original
        finally:
            # Defensive: restore INFO in case the test fails mid-loop and
            # leaves the module state polluted for subsequent tests.
            INFO["state_description"] = original
