import unittest
import os
import json
import hashlib

import numpy as np
from dacbench import AbstractEnv
from dacbench.envs.sgd import SGDEnv, Reward
from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.benchmarks.sgd_benchmark import SGDBenchmark, SGD_DEFAULTS
from dacbench.wrappers import ObservationWrapper


class TestSGDEnv(unittest.TestCase):
    def setUp(self):
        bench = SGDBenchmark()
        self.env = bench.get_benchmark(seed=123)

    @staticmethod
    def data_path(path):
        return os.path.join(os.path.dirname(__file__), 'data', path)

    def test_setup(self):
        self.assertTrue(issubclass(type(self.env), AbstractEnv))
        self.assertFalse(self.env.no_cuda)
        self.assertTrue(self.env.model is None)
        self.assertTrue(self.env.current_training_loss is None)
        self.assertTrue(self.env.batch_size == SGD_DEFAULTS["training_batch_size"])
        self.assertTrue(self.env.initial_lr == self.env.current_lr)

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
            state, reward, done, meta = env.step(1.0)
            self.assertTrue(reward >= env.reward_range[0])
            self.assertTrue(reward <= env.reward_range[1])
            self.assertFalse(done)
            self.assertTrue(len(meta.keys()) == 0)

    def test_get_default_state(self):
        self.env.reset()
        state, _, _, _ = self.env.step(0.5)
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
                    "alignment"
                ],
            )
        )
        self.assertTrue(state["currentLR"] == 0.5)
        self.assertTrue(state["trainingLoss"] > 0)
        self.assertTrue(state["validationLoss"] > 0)

    def test_hash(self):
        string_config = {k: str(v) for (k, v) in SGD_DEFAULTS.items()}
        h = hashlib.sha1(json.dumps(string_config).encode()).hexdigest()
        with open(self.data_path('sgd_config.hash'), 'r') as f:
            self.assertEqual(h, f.read())

    @staticmethod
    def replay(env, prev_mem):
        env = ObservationWrapper(env)
        env.reset()
        mem = []
        for x in prev_mem:
            action = x[-1]
            state, reward, done, _ = env.step(action)
            mem.append(np.concatenate([state, [reward, int(done), action]]))
        return np.array(mem)

    def test_functional_static(self):
        prev_mem = np.load(self.data_path('sgd_static_test.npy'))
        mem = self.replay(self.env, prev_mem)
        np.testing.assert_allclose(prev_mem, mem)

    def test_functional_dynamic(self):
        prev_mem = np.load(self.data_path('sgd_dynamic_test.npy'))
        mem = self.replay(self.env, prev_mem)
        np.testing.assert_allclose(prev_mem, mem)

    def test_close(self):
        self.assertTrue(self.env.close())

    def test_render(self):
        self.env.render("human")
        with self.assertRaises(NotImplementedError):
            self.env.render("random")
