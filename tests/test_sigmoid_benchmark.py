import unittest
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

    def test_read_instances(self):
        bench = SigmoidBenchmark()
        bench.read_instance_set()
        self.assertTrue(len(bench.config.instance_set)==100)
        self.assertTrue(len(bench.config.instance_set[0])==2)
        self.assertTrue(bench.config.instance_set[0]==[2.0004403531465558,7.903476325943215])
        bench2 = SigmoidBenchmark()
        env = bench2.get_benchmark_env()
        self.assertTrue(len(env.instance_set[0])==2)
        self.assertTrue(env.instance_set[0]==[2.0004403531465558,7.903476325943215])
        self.assertTrue(len(env.instance_set)==100)

    def test_benchmark_env(self):
        bench = SigmoidBenchmark()
        env = bench.get_complete_benchmark()
        self.assertTrue(issubclass(type(env), InstanceSamplingWrapper))

    def test_action_value_setting(self):
        bench = SigmoidBenchmark()
        bench.set_action_values([1, 2, 3])
        self.assertTrue(bench.config.action_values == [1, 2, 3])
        self.assertTrue(bench.config.action_space_args == [6])
        self.assertTrue(len(bench.config.observation_space_args[0]) == 10)
