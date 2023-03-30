import unittest

from dacbench import AbstractEnv
from dacbench.benchmarks.fast_downward_benchmark import FastDownwardBenchmark


class TestFDEnv(unittest.TestCase):
    def make_env(self):
        bench = FastDownwardBenchmark()
        env = bench.get_environment()
        return env

    def test_setup(self):
        env = self.make_env()
        self.assertTrue(issubclass(type(env), AbstractEnv))

    def test_reset(self):
        env = self.make_env()
        env.reset()
        self.assertFalse(env.socket is None)
        self.assertFalse(env.fd is None)

    def test_step(self):
        env = self.make_env()
        env.reset()
        state, reward, terminated, truncated, meta = env.step(1)
        self.assertTrue(reward >= env.reward_range[0])
        self.assertTrue(reward <= env.reward_range[1])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(len(meta.keys()) == 0)
        self.assertTrue(len(state) == 10)

    def test_close(self):
        env = self.make_env()
        self.assertTrue(env.close())
        self.assertTrue(env.conn is None)
        self.assertTrue(env.socket is None)

    def test_render(self):
        env = self.make_env()
        env.render()
