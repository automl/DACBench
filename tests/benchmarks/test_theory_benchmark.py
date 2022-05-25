from dacbench.benchmarks import TheoryBenchmark
from dacbench.envs import RLSEnv, RLSEnvDiscrete
import unittest
import json
import os


class TestTheoryBenchmark(unittest.TestCase):
    def test_get_env(self):
        # environment with non-discrete action space
        bench = TheoryBenchmark(
            config={"discrete_action": False, "min_action": 1, "max_action": 49}
        )
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), RLSEnv))

        # environment with discrete action space
        bench = TheoryBenchmark(
            config={"discrete_action": True, "action_choices": [1, 2, 4, 8]}
        )
        env = bench.get_environment()
        self.assertTrue(issubclass(type(env), RLSEnvDiscrete))

    def test_save_conf(self):
        # save a benchmark config
        bench = TheoryBenchmark()
        bench.save_config("test_conf.json")

        # reload it
        with open("test_conf.json", "r") as f:
            recovered = json.load(f)
        os.remove("test_conf.json")

        # create a new benchmark with the loaded config
        rbench = TheoryBenchmark(config=recovered)

        # check if the two benchmarks are identical
        assert isinstance(rbench.get_environment(), RLSEnvDiscrete)
        for i in range(len(bench.config.action_choices)):
            assert bench.config.action_choices[i] == rbench.config.action_choices[i]


if __name__ == "__main__":
    TestTheoryBenchmark().test_get_env()
    TestTheoryBenchmark().test_save_conf()
