from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import FastDownwardEnv

import numpy as np
import os

NUM_HEURISTICS = 2

FD_DEFAULTS = objdict(
    {
        "num_heuristics": NUM_HEURISTICS,
        "action_space_class": "Discrete",
        "action_space_args": [NUM_HEURISTICS],
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [
            np.array([-np.inf for _ in range(5 * NUM_HEURISTICS)]),
            np.array([np.inf for _ in range(5 * NUM_HEURISTICS)]),
        ],
        "reward_range": (-np.inf, 0),
        "cutoff": 1e6,
        "use_general_state_info": True,
        "host": "",
        "port": 52322,
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
        + "/../envs/fast-downward/fast-downward.py",
        "parallel": True,
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

        return FastDownwardEnv(self.config)

    def read_instance_set(self):
        """
        Read paths of instances from config into list
        """
        instances = []
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/"
            + self.config.instance_set_path
        )
        for root, dirs, files in os.walk(path):
            for file in files:
                if (
                    file.endswith(".pddl") or file.endswith(".sas")
                ) and not file.startswith("domain"):
                    instances.append(os.path.join(root, file))
        self.config["instance_set"] = instances

        if instances[0].endswith(".pddl"):
            self.config.domain_file = self.config.instance_set_path + "/domain.pddl"

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
