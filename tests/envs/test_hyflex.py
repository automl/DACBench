import pytest
import unittest
import numpy as np
from dacbench import AbstractEnv
from dacbench.benchmarks import HyFlexBenchmark


class TestHyFlexEnv(unittest.TestCase):
    def make_env(self):
        bench = HyFlexBenchmark()
        env = bench.get_environment()
        return env

    def test_setup(self):
        env = self.make_env()
        self.assertTrue(issubclass(type(env), AbstractEnv))
        for var_name in ['mem_size','s_best','s_inc','s_prop','reject','accept']:
            self.assertFalse(getattr(env, var_name) is None)
        for var_name in ['problem','unary_heuristics','binary_heuristics','f_best','f_prop','f_inc']:
            self.assertTrue(getattr(env, var_name) is None)

    def test_reset(self):
        env = self.make_env()
        env.reset()
        for var_name in ['problem','unary_heuristics','binary_heuristics','f_best','f_prop','f_inc']:
            self.assertFalse(getattr(env, var_name) is None)

    def test_step(self):
        env = self.make_env()
        env.reset()
        for i in range(10):
            cur_f_best = env.f_best        
            state, reward, done, _ = env.step(env.action_space.sample())                    
            self.assertTrue(reward >= env.reward_range[0])        
            self.assertTrue(reward <= env.reward_range[1])
            self.assertTrue(env.f_best <= cur_f_best)
        