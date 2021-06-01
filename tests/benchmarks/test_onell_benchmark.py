import unittest
import json
import os

from dacbench.benchmarks import OneLLBenchmark
from dacbench.envs import OneLLEnv


class TestOneLLBenchmark(unittest.TestCase):
    def test_get_env(self):
        bench = OneLLBenchmark()
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), OneLLEnv))

    def test_scenarios(self):
        scenarios = ["lbd_theory", "lbd_onefifth", "lbd_p_c", "lbd1_lbd2_p_c"]
        for s in scenarios:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../dacbench/additional_configs/onell",
                s + ".json",
            )
            bench = OneLLBenchmark(path)
            self.assertTrue(bench.config is not None)
            env = bench.get_environment()
            state = env.reset()
            self.assertTrue(state is not None)
            action = env.action_space.sample()
            state, _, _, _ = env.step(action)
            self.assertTrue(state is not None)

    def test_save_conf(self):
        bench = OneLLBenchmark()
        bench.save_config("test_conf.json")
        with open("test_conf.json", "r") as fp:
            recovered = json.load(fp)

        # instance_set is currently dropped, since storing the instance_set in not possible for all benchmarks.
        # see (https://github.com/automl/DACBench/issues/87)
        key_difference = set(bench.config.keys()).difference(set(recovered.keys()))
        self.assertSetEqual(key_difference, {"instance_set"})

        os.remove("test_conf.json")

    def test_read_instances(self):
        bench = OneLLBenchmark()
        bench.read_instance_set()
        for name in ["max_evals", "size"]:
            self.assertTrue(name in bench.config.instance_set[0])
        first_inst = bench.config.instance_set[0]

        bench2 = OneLLBenchmark()
        env = bench2.get_environment()
        self.assertTrue(env.instance_set[0] == first_inst)


# TestOneLLBenchmark().test_get_env()
# TestOneLLBenchmark().test_scenarios()
# TestOneLLBenchmark().test_read_instances()
# TestOneLLBenchmark().test_save_conf()
