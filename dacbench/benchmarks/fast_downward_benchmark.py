import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import FastDownwardEnv

HEURISTICS = [
    "tiebreaking([pdb(pattern=manual_pattern([0,1])),weight(g(),-1)])",
    "tiebreaking([pdb(pattern=manual_pattern([0,2])),weight(g(),-1)])",
]

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
HEURISTIC = CSH.CategoricalHyperparameter(name="heuristic", choices=["toy1", "toy2"])
DEFAULT_CFG_SPACE.add_hyperparameter(HEURISTIC)

INFO = {
    "identifier": "FastDownward",
    "name": "Heuristic Selection for the FastDownward Planner",
    "reward": "Negative Runtime (-1 per step)",
    "state_description": [
        "Average Value (heuristic 1)",
        "Max Value (heuristic 1)",
        "Min Value (heuristic 1)",
        "Open List Entries (heuristic 1)",
        "Variance (heuristic 1)",
        "Average Value (heuristic 2)",
        "Max Value (heuristic 2)",
        "Min Value (heuristic 2)",
        "Open List Entries (heuristic 2)",
        "Variance (heuristic 2)",
    ],
}

FD_DEFAULTS = objdict(
    {
        "heuristics": HEURISTICS,
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [
            np.array([-np.inf for _ in range(5 * len(HEURISTICS))]),
            np.array([np.inf for _ in range(5 * len(HEURISTICS))]),
        ],
        "reward_range": (-np.inf, 0),
        "cutoff": 1e6,
        "use_general_state_info": True,
        "host": "",
        "port": 54322,
        "control_interval": 0,
        "fd_seed": 0,
        "num_steps": None,
        "state_type": 2,
        "config_dir": ".",
        "port_file_id": None,
        "seed": 0,
        "max_rand_steps": 0,
        "instance_set_path": "../instance_sets/fast_downward/train",
        "test_set_path": "../instance_sets/fast_downward/test",
        "fd_path": os.path.dirname(os.path.abspath(__file__))
        + "/../envs/rl-plan/fast-downward/fast-downward.py",
        "parallel": True,
        "fd_logs": None,
        "benchmark_info": INFO,
    }
)


class FastDownwardBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for Sigmoid
    """

    def __init__(self, config_path=None, config=None):
        """
        Initialize FD Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(FastDownwardBenchmark, self).__init__(config_path, config)
        if not self.config:
            self.config = objdict(FD_DEFAULTS.copy())

        for key in FD_DEFAULTS:
            if key not in self.config:
                self.config[key] = FD_DEFAULTS[key]

    def get_environment(self):
        """
        Return Luby env with current configuration

        Returns
        -------
        LubyEnv
            Luby environment
        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        # Read test set if path is specified
        if (
            "test_set" not in self.config.keys()
            and "test_set_path" in self.config.keys()
        ):
            self.read_instance_set(test=True)

        env = FastDownwardEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self, test=False):
        """
        Read paths of instances from config into list
        """
        instances = {}
        if test:
            path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/"
                + self.config.test_set_path
            )
            keyword = "test_set"
        else:
            path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/"
                + self.config.instance_set_path
            )
            keyword = "instance_set"
        import re

        for root, dirs, files in os.walk(path):
            for f in files:
                if (f.endswith(".pddl") or f.endswith(".sas")) and not f.startswith(
                    "domain"
                ):
                    p = os.path.join(root, f)
                    if f.endswith(".pddl"):
                        index = p.split("/")[-1].split(".")[0]
                    else:
                        index = p.split("/")[-2]
                    index = int(re.sub("[^0-9]", "", index))
                    instances[index] = p
        if len(instances) == 0:
            for f in os.listdir(path):
                f = f.strip()
                if (f.endswith(".pddl") or f.endswith(".sas")) and not f.startswith(
                    "domain"
                ):
                    p = os.path.join(path, f)
                    if f.endswith(".pddl"):
                        index = p.split("/")[-1].split(".")[0]
                    else:
                        index = p.split("/")[-2]
                    index = re.sub("[^0-9]", "", index)
                    instances[index] = p
        self.config[keyword] = instances

        if instances[list(instances.keys())[0]].endswith(".pddl"):
            self.config.domain_file = os.path.join(path + "/domain.pddl")

    def set_heuristics(self, heuristics):
        self.config.heuristics = heuristics
        self.config.action_space_args = [len(heuristics)]
        self.config.observation_space_args = [
            np.array([-np.inf for _ in range(5 * len(heuristics))]),
            np.array([np.inf for _ in range(5 * len(heuristics))]),
        ]

    def get_benchmark(self, seed=0):
        """
        Get published benchmark

        Parameters
        -------
        seed : int
            Environment seed

        Returns
        -------
        env : FastDownwardEnv
            FD environment
        """
        self.config = objdict(FD_DEFAULTS.copy())
        self.read_instance_set()
        self.read_instance_set(test=True)
        self.config.seed = seed
        env = FastDownwardEnv(self.config)
        return env
