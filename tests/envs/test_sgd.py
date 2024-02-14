import os
import unittest
import torch
from unittest.mock import patch

import numpy as np

from dacbench import AbstractEnv
from dacbench.abstract_benchmark import objdict
from dacbench.benchmarks.sgd_benchmark import SGD_DEFAULTS, SGDBenchmark
from dacbench.envs.sgd_new import SGDEnv
from dacbench.wrappers import ObservationWrapper


class TestSGDEnv(unittest.TestCase):
    def make_env(self):
        self.seed = 111
        bench = SGDBenchmark()
        env = bench.get_benchmark(seed=self.seed)
        return env

    @staticmethod
    def data_path(path):
        return os.path.join(os.path.dirname(__file__), "data", path)

    def test_setup(self):
        env = self.make_env()
        self.assertTrue(issubclass(type(env), AbstractEnv))
        self.assertEqual(env.learning_rate, None)
        self.assertEqual(env.initial_seed, self.seed)
        self.assertTrue(env.batch_size == SGD_DEFAULTS["training_batch_size"])
        self.assertTrue(env.use_momentum == SGD_DEFAULTS["use_momentum"])
        self.assertEqual(env.n_steps, SGD_DEFAULTS["cutoff"])

    def test_reward_function(self):
        def dummy_func(x):
            return 4 * x + 4

        benchmark = SGDBenchmark()
        benchmark.config = objdict(SGD_DEFAULTS.copy())
        benchmark.config["reward_function"] = dummy_func
        benchmark.read_instance_set()

        env = SGDEnv(benchmark.config)
        self.assertEqual(env.optimizer_params, SGD_DEFAULTS.optimizer_params)
        self.assertEqual(env.get_reward, dummy_func)

        env2 = self.make_env()
        self.assertEqual(env2.get_reward, env2.get_default_reward)

    def test_reset(self):
        env = self.make_env()
        state, info = env.reset()
        self.assertIsInstance(state, dict)
        self.assertIsInstance(info, dict)
        self.assertTrue(env.loss == 0)
        self.assertEqual(env._done, False)

    def test_step(self):
        env = self.make_env()
        state, info = env.reset()

        # Test if step method executes without error
        state, reward, done, truncated, info = env.step(0.001)
        self.assertIsInstance(state, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        if not env._done:
            self.assertTrue(env.min_validation_loss is not None)

    def test_crash(self):
        with patch("dacbench.envs.sgd_new.forward_backward") as mock_forward_backward:
            mock_forward_backward.return_value = torch.tensor(float("inf"))
            env = ObservationWrapper(self.make_env())
            state, info = env.reset()

            state, reward, terminated, truncated, info = env.step(np.nan)
            self.assertTrue(env._done)
            self.assertEqual(reward, env.crash_penalty)
            self.assertFalse(terminated)
            self.assertTrue(truncated)

    def test_reproducibility(self):
        mems = []
        instances = []
        env = self.make_env()

        for _ in range(2):
            rng = np.random.default_rng(123)
            env.seed(123)
            env.instance_index = 0
            instances.append(env.get_instance_set())

            state, info = env.reset()

            terminated, truncated = False, False
            mem = []
            step = 0
            while not (terminated or truncated) and step < 5:
                action = np.exp(rng.integers(low=-10, high=1))
                state, reward, terminated, truncated, _ = env.step(action)
                mem.append([state, [reward, int(truncated), action]])
                step += 1
            mems.append(mem)
        self.assertEqual(len(mems[0]), len(mems[1]))
        self.assertEqual(instances[0], instances[1])

    def test_close(self):
        env = self.make_env()
        self.assertTrue(env.close() is None)

    def test_render(self):
        env = self.make_env()
        state, info = env.reset()
        env.render("human")
        with self.assertRaises(NotImplementedError):
            env.render("random")
