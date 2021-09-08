from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import HyFlexEnv
from gym import spaces
import numpy as np
import sys
import os
import csv

DOMAINS = {
    "Toy": {"n_heuristics": 5, "n_instances": 12},
    "SAT": {"n_heuristics": 10, "n_instances": 12},
    "BinPacking": {"n_heuristics": 9, "n_instances": 12},
    "FlowShop": {"n_heuristics": 15, "n_instances": 12},
    "PersonnelScheduling": {"n_heuristics": 12, "n_instances": 12},
    "TSP": {"n_heuristics": 16, "n_instances": 10},
    "VRP": {"n_heuristics": 13, "n_instances": 10}
}

INPUT_DIM = 10

INFO = {"identifier": "hyflex",
        "name": "Selection hyper-heuristic",
        "reward": "Improvement on Best Fitness (f_prev_best - f_new_best)",
        "state_description": ["Fitness Delta (f_incumbent - f_proposed)"]}

HYFLEX_DEFAULTS = objdict(
    {
        "learn_select": False,
        "learn_accept": True,
        "action_space_class": None,  # Set automatically in read_instance_set
        "action_space_args": None,  # Set automatically in read_instance_set
        "observation_space_class": "Dict",
        "observation_space_type": None,
        "observation_space_args": [
            {
                "f_delta": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,)
                )
            }
        ],
        "reward_range": (0, sys.float_info.max),
        "cutoff": 1e3,
        "seed": 42,
        "instance_set_path": "../instance_sets/hyflex/chesc.csv",
        "benchmark_info": INFO
    }
)


class HyFlexBenchmark(AbstractBenchmark):
    """
    Benchmark for learning a selection hyper-heuristic for HyFlex
    """

    def __init__(self, config_path=None, **kwargs):
        """
        Initialize CMA Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(HyFlexBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = objdict(HYFLEX_DEFAULTS.copy())

        for key in HYFLEX_DEFAULTS:
            if key not in self.config:
                self.config[key] = HYFLEX_DEFAULTS[key]

        for key in kwargs:
            assert(key in HYFLEX_DEFAULTS)
            self.config[key] = kwargs[key]

        assert(self.config["learn_select"] or self.config["learn_accept"])

    def get_environment(self):
        """
        Return HyFlexEnv env with current configuration

        Returns
        -------
        HyFlexEnv
            HyFlex environment
        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        env = HyFlexEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def get_benchmark(self, seed=0):
        self.config.seed = seed
        return self.get_environment()
    
    def read_instance_set(self):
        """
        Read path of instances from config into list
        """
        path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/"
                + self.config.instance_set_path
        )
        self.config["instance_set"] = {}
        with open(path, "r") as fh:
            reader = csv.DictReader(fh)
            single_domain = None
            for row in reader:
                instance_index = int(row["instance"])
                instance = [
                    row["domain"],
                    int(row["instance"]),
                    int(row["seed"]) if "seed" in row else self.config["seed"],
                    int(row["cutoff"]) if "cutoff" in row else self.config["cutoff"],
                ]
                self.config["instance_set"][int(row["ID"])] = instance
                assert(row["domain"] in DOMAINS)
                assert(0 <= int(row["instance"]) < DOMAINS[row["domain"]]["n_instances"])
                if self.config["learn_select"]:
                    if single_domain is None:
                        single_domain = row["domain"]
                    else:
                        # When self.config["learn_select"] == True, we only support instances from a single domain
                        assert(row["domain"] == single_domain)

        # configure action space
        if self.config["learn_select"] and self.config["learn_accept"]:
            self.config["action_space_class"] = "MultiDiscrete"
            self.config["action_space_args"] = [[DOMAINS[single_domain]["n_heuristics"], 2]]
        else:
            self.config["action_space_class"] = "Discrete"
            if self.config["learn_select"]:
                self.config["action_space_args"] = [DOMAINS[single_domain]["n_heuristics"]]
            else:
                self.config["action_space_args"] = [2]


if __name__ == "__main__":
    #bench = HyFlexBenchmark(instance_set_path="../instance_sets/hyflex/sat3.csv", cutoff=10000)
    bench = HyFlexBenchmark(instance_set_path="../instance_sets/hyflex/toy_train.csv", cutoff=100,
                            learn_select=True, learn_accept=True)
    env = bench.get_benchmark()

    done = False
    obs = env.reset()
    while not done:
        print(obs)
        obs, reward, done, info = env.step([1, 3])
        print(info)
    print(obs)
