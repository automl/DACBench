from __future__ import annotations

import os
import unittest

import numpy as np
from dacbench import AbstractEnv
from dacbench.abstract_benchmark import objdict
from dacbench.benchmarks import GeometricBenchmark
from dacbench.envs import GeometricEnv

FILE_PATH = os.path.dirname(__file__)

DEFAULTS_STATIC = objdict(
    {
        "action_space_class": "Discrete",
        "action_space_args": [],
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [],
        "reward_range": (0, 1),
        "cutoff": 10,
        "action_values": [],
        "action_value_default": 4,
        "action_values_variable": False,  # if True action value mapping will be used
        "action_interval_mapping": {},  # maps actions to equally sized intervalls in [-1, 1]  # clip function value if it is higher than this number
        "derivative_interval": 3,
        "realistic_trajectory": True,
        "instance_set_path": "../instance_sets/geometric/geometric_test.csv",
        "correlation_table": None,
        "correlation_info": {
            "high": [(1, 2, "+"), (2, 3, "-"), (1, 5, "+")],
            "middle": [(4, 5, "-")],
            "low": [(4, 7, "+"), (2, 3, "+"), (0, 2, "-")],
        },
        "correlation_mapping": {
            "high": (0.5, 1),
            "middle": (0.1, 0.5),
            "low": (0, 0.1),
        },
        "correlation_depth": 4,
        "correlation_active": True,
        "benchmark_info": "Hallo",
    }
)


class TestGeometricEnv(unittest.TestCase):
    def make_env(self, config: dict):
        geo_bench = GeometricBenchmark()
        geo_bench.read_instance_set()
        geo_bench.set_action_values()
        geo_bench.create_correlation_table()

        config["action_interval_mapping"] = geo_bench.config.action_interval_mapping
        config["instance_set"] = geo_bench.config.instance_set
        config["action_values"] = geo_bench.config.action_values
        config["config_space"] = geo_bench.config.config_space
        config["observation_space_args"] = geo_bench.config.observation_space_args
        config["correlation_table"] = geo_bench.config.correlation_table
        config["correlation_active"] = True

        return GeometricEnv(config)

    def test_setup(self):
        env = self.make_env(DEFAULTS_STATIC)
        assert issubclass(type(env), AbstractEnv)
        assert env.np_random is not None
        assert env.n_steps == 10
        assert env.n_actions == len(env.action_vals)
        assert isinstance(env.action_interval_mapping, dict)

    def test_reset(self):
        env = self.make_env(DEFAULTS_STATIC)
        state, info = env.reset()
        assert state[0] == DEFAULTS_STATIC["cutoff"]
        assert issubclass(type(info), dict)
        assert not env._prev_state
        assert isinstance(env.action_trajectory, list)
        assert isinstance(env.action_trajectory_set, dict)

    def test_step(self):
        env = self.make_env(DEFAULTS_STATIC)
        env.reset()
        state, reward, terminated, truncated, meta = env.step(env.action_space.sample())
        assert reward >= env.reward_range[0]
        assert reward <= env.reward_range[1]
        assert state[0] == 9
        assert isinstance(state, np.ndarray)
        assert len(state) == 2 + 2 * env.n_actions
        assert not terminated
        assert not truncated
        assert len(meta.keys()) == 0

    def test_close(self):
        env = self.make_env(DEFAULTS_STATIC)
        assert env.close()

    def test_functions(self):
        env = self.make_env(DEFAULTS_STATIC)
        functions = env.functions
        assert functions._sigmoid(1, 0, 0) == 0.5
        assert functions._linear(5, 2, -3) == 7
        assert functions._constant(5) == 5
        self.assertAlmostEqual(functions._logarithmic(2, 2), 1.39, places=2)
        self.assertAlmostEqual(functions._sinus(4, 0.5), 0.91, places=2)

    def test_calculate_norm_values(self):
        env = self.make_env(DEFAULTS_STATIC)
        env.functions.calculate_norm_values(env.instance_set)
        assert env.functions.norm_calculated

    def test_calculate_function_value(self):
        env = self.make_env(DEFAULTS_STATIC)
        env.functions.instance_idx = 2
        env.functions.norm_calculated = False
        function_info = [2, "linear", 1, 2]
        assert env.functions._calculate_function_value(0, function_info, 0) == 2.0

    def test_calculate_derivative(self):
        env = self.make_env(DEFAULTS_STATIC)
        trajectory1 = [np.zeros(env.n_actions)]
        assert (
            env.functions.calculate_derivative(trajectory1, env.c_step)
            == np.zeros(env.n_actions)
        ).all()
        env.c_step = 1
        trajectory2 = [np.zeros(env.n_actions), np.ones(env.n_actions)]
        assert (
            env.functions.calculate_derivative(trajectory2, env.c_step)
            == np.ones(env.n_actions)
        ).all()

        trajectory2 = [
            np.zeros(env.n_actions),
            np.ones(env.n_actions),
            np.ones(env.n_actions) * 2,
        ]
        env.c_step = 2
        assert (
            env.functions.calculate_derivative(trajectory2, env.c_step)
            == np.ones(env.n_actions)
        ).all()

        trajectory3 = [
            np.zeros(env.n_actions),
            np.ones(env.n_actions),
            np.ones(env.n_actions) * 2,
            np.ones(env.n_actions) * 4,
            np.ones(env.n_actions) * 7,
        ]
        env.c_step = 4
        assert (
            env.functions.calculate_derivative(trajectory3, env.c_step)
            == np.ones(env.n_actions) * 2
        ).all()

    def test_get_coordinates_at_time_step(self):
        env = self.make_env(DEFAULTS_STATIC)
        assert (
            len(env.functions.get_coordinates_at_time_step(env.c_step)) == env.n_actions
        )

    def test_get_optimal_policy(self):
        env = self.make_env(DEFAULTS_STATIC)

        assert env.get_optimal_policy().shape == (env.n_steps, env.n_actions)
        assert len(env.get_optimal_policy(vector_action=False)) == env.n_steps

    def test_render_dimensions(self):
        env = self.make_env(DEFAULTS_STATIC)
        dimensions = [1, 2]
        env.render(dimensions, FILE_PATH)
        fig_title = f"GeoBench-Dimensions{len(dimensions)}.jpg"
        assert os.path.exists(os.path.join(FILE_PATH, fig_title))
        os.remove(os.path.join(FILE_PATH, fig_title))

    def test_render_3d_dimensions(self):
        env = self.make_env(DEFAULTS_STATIC)
        env.render_3d_dimensions([0, 1], FILE_PATH)
        assert os.path.exists(os.path.join(FILE_PATH, "3D.jpg"))
        assert os.path.exists(os.path.join(FILE_PATH, "3D-90side.jpg"))
        os.remove(os.path.join(FILE_PATH, "3D.jpg"))
        os.remove(os.path.join(FILE_PATH, "3D-90side.jpg"))
