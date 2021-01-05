import unittest
import json
import os

from dacbench.benchmarks import SigmoidBenchmark
from dacbench.envs import SigmoidEnv
from dacbench.wrappers import InstanceSamplingWrapper


class TestSigmoidBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = SigmoidBenchmark()
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), SigmoidEnv))

    def test_scenarios(self):
        scenarios = [
            "sigmoid_1D3M.json",
            "sigmoid_2D3M.json",
            "sigmoid_3D3M.json",
            "sigmoid_5D3M.json",
        ]
        for s in scenarios:
            path = os.path.join("dacbench/scenarios/sigmoid", s)
            bench = SigmoidBenchmark(path)
            self.assertTrue(bench.config is not None)
            env = bench.get_environment()
            state = env.reset()
            self.assertTrue(state is not None)
            state, _, _, _ = env.step(0)
            self.assertTrue(state is not None)

    def test_save_conf(self):
        bench = SigmoidBenchmark()
        bench.save_config("test_conf.json")
        with open("test_conf.json", "r") as fp:
            recovered = json.load(fp)
        for k in bench.config.keys():
            self.assertTrue(k in recovered.keys())
        os.remove("test_conf.json")

    def test_read_instances(self):
        bench = SigmoidBenchmark()
        bench.read_instance_set()
        self.assertTrue(len(bench.config.instance_set.keys()) == 100)
        self.assertTrue(len(bench.config.instance_set[0]) == 2)
        self.assertTrue(
            bench.config.instance_set[0] == [2.0004403531465558, 7.903476325943215]
        )
        bench2 = SigmoidBenchmark()
        env = bench2.get_environment()
        self.assertTrue(len(env.instance_set[0]) == 2)
        self.assertTrue(env.instance_set[0] == [2.0004403531465558, 7.903476325943215])
        self.assertTrue(len(env.instance_set.keys()) == 100)

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
