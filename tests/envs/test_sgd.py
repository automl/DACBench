import os
import unittest

import numpy as np

from dacbench import AbstractEnv
from dacbench.abstract_benchmark import objdict
from dacbench.benchmarks.sgd_benchmark import SGD_DEFAULTS, SGDBenchmark
from dacbench.envs.sgd import Reward, SGDEnv
from dacbench.wrappers import ObservationWrapper


class TestSGDEnv(unittest.TestCase):
    def setUp(self):
        bench = SGDBenchmark()
        self.env = bench.get_benchmark(seed=123)

    @staticmethod
    def data_path(path):
        return os.path.join(os.path.dirname(__file__), "data", path)

    def test_setup(self):
        self.assertTrue(issubclass(type(self.env), AbstractEnv))
        self.assertFalse(self.env.no_cuda)
        self.assertTrue(self.env.model is None)
        self.assertTrue(self.env.current_training_loss is None)
        self.assertTrue(self.env.batch_size == SGD_DEFAULTS["training_batch_size"])
        self.assertTrue(self.env.initial_lr == self.env.current_lr)

    def test_reward_type(self):
        benchmark = SGDBenchmark()
        benchmark.config = objdict(SGD_DEFAULTS.copy())
        benchmark.read_instance_set()

        env = SGDEnv(benchmark.config)
        self.assertEqual(env.reward_type, SGD_DEFAULTS.reward_type)

        benchmark.config.reward_type = SGD_DEFAULTS.reward_type.name
        env = SGDEnv(benchmark.config)
        self.assertEqual(env.reward_type, SGD_DEFAULTS.reward_type)

        benchmark.config.reward_type = "invalid_reward"
        with self.assertRaises(ValueError):
            env = SGDEnv(benchmark.config)

        benchmark.config.reward_type = 0
        with self.assertRaises(ValueError):
            env = SGDEnv(benchmark.config)

    def test_reset(self):
        self.env.reset()
        self.assertFalse(self.env.model is None)
        self.assertFalse(self.env.train_dataset is None)
        self.assertFalse(self.env.validation_dataset is None)

    def test_step(self):
        benchmark = SGDBenchmark()
        benchmark.config = objdict(SGD_DEFAULTS.copy())
        benchmark.read_instance_set()

        for reward_type in Reward:
            benchmark.config.reward_type = reward_type
            env = SGDEnv(benchmark.config)
            env = ObservationWrapper(env)
            self.assertTrue(env.reward_range == reward_type.func.frange)

            env.reset()
            _, reward, terminated, truncated, meta = env.step(1.0)
            self.assertTrue(reward >= env.reward_range[0])
            self.assertTrue(reward <= env.reward_range[1])
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(len(meta.keys()) == 0)

    def test_crash(self):
        env = ObservationWrapper(self.env)
        env.reset()
        state, reward, terminated, truncated, _ = env.step(np.nan)
        self.assertTrue(env.crashed)
        self.assertFalse(any(np.isnan(state)))
        self.assertTrue(reward == env.crash_penalty)

    def test_stateless(self):
        env = ObservationWrapper(self.env)
        rng = np.random.default_rng(123)
        mems = []
        instance_idxs = []
        for _ in range(3):
            env.reset()
            instance_idxs.append(env.instance_index)

            terminated, truncated = False, False
            mem = []
            step = 0
            while not (terminated or truncated) and step < 5:
                action = np.exp(rng.integers(low=-10, high=1))
                state, reward, terminated, truncated, _ = env.step(action)
                mem.append(np.concatenate([state, [reward, int(truncated), action]]))
                step += 1
            mems.append(np.array(mem))

        rng = np.random.default_rng(123)
        for i, idx in enumerate(reversed(instance_idxs)):
            env.instance_index = idx - 1
            env.reset()
            self.assertTrue(env.instance_index == idx)

            terminated, truncated = False, False
            mem = []
            step = 0
            while not (terminated or truncated) and step < 5:
                action = mems[-(i + 1)][step][-1]
                state, reward, terminated, truncated, _ = env.step(action)
                mem.append(np.concatenate([state, [reward, int(truncated), action]]))
                step += 1
            np.testing.assert_allclose(mems[-(i + 1)], np.array(mem))

    def test_reproducibility(self):
        mems = []
        instances = []
        env = ObservationWrapper(self.env)
        for _ in range(2):
            rng = np.random.default_rng(123)
            env.seed(123)
            env.instance_index = 0
            instances.append(env.get_instance_set())

            env.reset()

            terminated, truncated = False, False
            mem = []
            step = 0
            while not (terminated or truncated) and step < 5:
                action = np.exp(rng.integers(low=-10, high=1))
                state, reward, terminated, truncated, _ = env.step(action)
                mem.append(np.concatenate([state, [reward, int(truncated), action]]))
                step += 1
            mems.append(np.array(mem))
        self.assertEqual(mems[0].size, mems[1].size)
        self.assertEqual(instances[0], instances[1])
        np.testing.assert_allclose(mems[0], mems[1])

    def test_get_default_state(self):
        self.env.reset()
        state, _, _, _, _ = self.env.step(0.5)
        self.assertTrue(issubclass(type(state), dict))
        self.assertTrue(
            np.array_equal(
                list(state.keys()),
                [
                    "predictiveChangeVarDiscountedAverage",
                    "predictiveChangeVarUncertainty",
                    "lossVarDiscountedAverage",
                    "lossVarUncertainty",
                    "currentLR",
                    "trainingLoss",
                    "validationLoss",
                    "step",
                    "alignment",
                    "crashed",
                ],
            )
        )
        self.assertTrue(state["currentLR"] == 0.5)
        self.assertTrue(state["trainingLoss"] > 0)
        self.assertTrue(state["validationLoss"] > 0)

    def test_close(self):
        self.assertTrue(self.env.close())

    def test_render(self):
        self.env.render("human")
        with self.assertRaises(NotImplementedError):
            self.env.render("random")
