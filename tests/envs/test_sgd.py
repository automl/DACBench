import pytest
import unittest
import numpy as np
from dacbench import AbstractEnv
from dacbench.benchmarks.sgd_benchmark import SGDBenchmark, SGD_DEFAULTS


class TestSGDEnv(unittest.TestCase):
    def make_env(self):
        bench = SGDBenchmark()
        env = bench.get_environment()
        return env

    def test_setup(self):
        env = self.make_env()
        self.assertTrue(issubclass(type(env), AbstractEnv))
        self.assertFalse(env.no_cuda)
        self.assertTrue(env.model is None)
        self.assertTrue(env.current_training_loss is None)
        self.assertTrue(env.batch_size == SGD_DEFAULTS["training_batch_size"])
        self.assertTrue(env.initial_lr == env.current_lr)

    def test_reset(self):
        env = self.make_env()
        env.reset()
        self.assertFalse(env.model is None)
        self.assertFalse(env.train_dataset is None)
        self.assertFalse(env.validation_dataset is None)

    def test_step(self):
        env = self.make_env()
        env.reset()
        state, reward, done, meta = env.step([1])
        self.assertTrue(reward >= env.reward_range[0])
        self.assertTrue(reward <= env.reward_range[1])
        self.assertFalse(done)
        self.assertTrue(len(meta.keys()) == 0)

    def test_get_default_state(self):
        env = self.make_env()
        env.reset()
        state, _, _, _ = env.step([0.5])
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
                ],
            )
        )
        self.assertTrue(state["currentLR"] == 10 ** -0.5)
        self.assertTrue(state["trainingLoss"] > 0)
        self.assertTrue(state["validationLoss"] > 0)

    def test_close(self):
        env = self.make_env()
        self.assertTrue(env.close())

    def test_render(self):
        env = self.make_env()
        env.render("human")
        with pytest.raises(NotImplementedError):
            env.render("random")
