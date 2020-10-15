import unittest
import json
import os

from daclib.benchmarks import SigmoidBenchmark
from daclib.envs import SigmoidEnv
from daclib.wrappers import InstanceSamplingWrapper


class TestSigmoidBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = SigmoidBenchmark()
        env = bench.get_benchmark_env()
        self.assertTrue(issubclass(type(env), SigmoidEnv))

    def test_setup(self):
        bench = SigmoidBenchmark()
        self.assertTrue(bench.config is not None)

        config = {"dummy": 0}
        with open("test_conf.json", "w+") as fp:
            json.dump(config, fp)
        bench = SigmoidBenchmark("test_conf.json")
        self.assertTrue(bench.config.dummy == 0)
        os.remove("test_conf.json")

    def test_read_instances(self):
        bench = SigmoidBenchmark()
        bench.read_instance_set()
        self.assertTrue(len(bench.config.instance_set) == 100)
        self.assertTrue(len(bench.config.instance_set[0]) == 2)
        self.assertTrue(
            bench.config.instance_set[0] == [2.0004403531465558, 7.903476325943215]
        )
        bench2 = SigmoidBenchmark()
        env = bench2.get_benchmark_env()
        self.assertTrue(len(env.instance_set[0]) == 2)
        self.assertTrue(env.instance_set[0] == [2.0004403531465558, 7.903476325943215])
        self.assertTrue(len(env.instance_set) == 100)

    def test_benchmark_env(self):
        bench = SigmoidBenchmark()

        for d in [1, 2, 3, 5]:
            env = bench.get_benchmark(d)
            self.assertTrue(issubclass(type(env), InstanceSamplingWrapper))
            env.reset()
            s, r, d, i = env.step(0)
            self.assertTrue(env.inst_id == 0)
            self.assertTrue(len(env.instance_set) == 1)

    def test_action_value_setting(self):
        bench = SigmoidBenchmark()
        bench.set_action_values([1, 2, 3])
        self.assertTrue(bench.config.action_values == [1, 2, 3])
        self.assertTrue(bench.config.action_space_args == [6])
        self.assertTrue(len(bench.config.observation_space_args[0]) == 10)
