import pickle
import unittest
import os

import numpy as np
from dacbench import AbstractEnv
from dacbench.envs.sgd import SGDEnv
from dacbench.benchmarks.sgd_benchmark import SGDBenchmark, SGD_DEFAULTS
from dacbench.wrappers import ObservationWrapper


class TestSGDEnv(unittest.TestCase):
    def setUp(self):
        bench = SGDBenchmark()
        self.env = bench.get_environment()
        self.data_path = os.path.join(os.path.dirname(__file__), 'data')

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
        self.env.reset()
        state, reward, done, meta = self.env.step(1.0)
        self.assertTrue(reward >= self.env.reward_range[0])
        self.assertTrue(reward <= self.env.reward_range[1])
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

    def test_functional(self):
        for test_case in [os.path.join(self.data_path, 'sgd_static_test.pickle'),
                          os.path.join(self.data_path, 'sgd_dynamic_test.pickle')]:
            with open(os.path.join(self.data_path, 'sgd_benchmark_config.pickle'), 'rb') as f:
                benchmark = SGDBenchmark()
                benchmark.config = pickle.load(f)
                env = SGDEnv(benchmark.config)
                env = ObservationWrapper(env)

            with open(test_case, 'rb') as f:
                prev_mem = pickle.load(f)

            env.reset()
            done = False
            mem = []
            step = 0
            while not done and step < 50:
                action = prev_mem[step][-1]
                state, reward, done, _ = env.step(action)
                mem.append(np.concatenate([state, [reward, int(done), action]]))
                step += 1

            np.testing.assert_allclose(prev_mem, np.array(mem))

    def test_close(self):
        self.assertTrue(self.env.close())

    def test_render(self):
        self.env.render("human")
        with self.assertRaises(NotImplementedError):
            self.env.render("random")
