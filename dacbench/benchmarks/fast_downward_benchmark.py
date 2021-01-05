from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import FastDownwardEnv

import numpy as np
import os

HEURISTICS = [
    "tiebreaking([pdb(pattern=manual_pattern([0,1])),weight(g(),-1)])",
    "tiebreaking([pdb(pattern=manual_pattern([0,2])),weight(g(),-1)])",
]

INFO = {"name": "Heuristic Selection for the FastDownward Planner",
        "reward": "Negative Runtime (-1 per step)",
        "state_description": ["Average Value (heuristic 1)",
                              "Max Value (heuristic 1)",
                              "Min Value (heuristic 1)",
                              "Open List Entries (heuristic 1)",
                              "Variance (heuristic 1)",
                              "Average Value (heuristic 2)",
                              "Max Value (heuristic 2)",
                              "Min Value (heuristic 2)",
                              "Open List Entries (heuristic 2)",
                              "Variance (heuristic 2)"
                              ]}

FD_DEFAULTS = objdict(
    {
        "heuristics": HEURISTICS,
        "action_space_class": "Discrete",
        "action_space_args": [len(HEURISTICS)],
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
        "fd_path": os.path.dirname(os.path.abspath(__file__))
        + "/../envs/rl-plan/fast-downward/fast-downward.py",
        "parallel": True,
        "fd_logs": None,
        "benchmark_info": INFO
    }
)


class FastDownwardBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for Sigmoid
    """

    def __init__(self, config_path=None):
        """
        Initialize FD Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(FastDownwardBenchmark, self).__init__(config_path)
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

        env = FastDownwardEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self):
        """
        Read paths of instances from config into list
        """
        instances = {}
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/"
            + self.config.instance_set_path
        )
        import re
        for root, dirs, files in os.walk(path):
            for f in files:
                if (f.endswith(".pddl") or f.endswith(".sas")) and not f.startswith(
                    "domain"
                ):
                    path = os.path.join(root, f)
                    index = path.split("/")[-1].split(".")[0]
                    index = re.sub("[^0-9]", "", index)
                    instances[index] = os.path.join(root, f)
        if len(instances) == 0:
            for f in os.listdir(path):
                f = f.strip()
                if (f.endswith(".pddl") or f.endswith(".sas")) and not f.startswith(
                    "domain"
                ):
                    path = os.path.join(path, f)
                    index = path.split("/")[-1].split(".")[0]
                    index = re.sub("[^0-9]", "", index)
                    instances[index] = path
        self.config["instance_set"] = instances

        if instances[0].endswith(".pddl"):
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
        self.config.seed = seed
        env = FastDownwardEnv(self.config)
        return env
