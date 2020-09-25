from daclib.abstract_benchmark import AbstractBenchmark, objdict
from daclib.envs import FastDownwardEnv

from gym import spaces
import numpy as np
import os

NUM_HEURISTICS = 2

FD_DEFAULTS = objdict(
    {
        "num_heuristics": NUM_HEURISTICS,
        "action_space": "Discrete",
        "action_space_args": [NUM_HEURISTICS],
        "observation_space": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [np.array([-np.inf for _ in range(5*NUM_HEURISTICS)]), np.array([np.inf for _ in range(5*NUM_HEURISTICS)])],
        "reward_range": (-np.inf, 0),
        "cutoff": 1e6,
        "use_general_state_info": True,
        "host": "",
        "port": 52322,
        "control_interval": 0,
        "fd_seed": 0,
        "num_steps": None,
        "state_type": 2,
        "config_dir": '.',
        "port_file_id": None,
        "seed": 0,
        "max_rand_steps": 0,
        "instance_set_path": "../instance_sets/fast_downward/train",
        "fd_path": os.path.dirname(os.path.abspath(__file__)) + "/../envs/fast-downward/fast-downward.py"
    }
)


class FastDownwardBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for Sigmoid
    """

    def __init__(self, config_path=None):
        super(FastDownwardBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = FD_DEFAULTS

        for key in FD_DEFAULTS:
            if not key in self.config:
                self.config[key] = FD_DEFAULTS[key]

        if not "instance_set" in self.config.keys():
            self.read_instance_set()

    def get_benchmark_env(self):
        """
        Return Luby env with current configuration

        Returns
        -------
        LubyEnv
            Luby environment
        """
        return FastDownwardEnv(self.config)

    def read_instance_set(self):
        """
        Read paths of instances from config into list
        """
        directory = self.config.instance_set_path
        instances = []
        path = os.path.dirname(os.path.abspath(__file__)) + "/" + self.config.instance_set_path
        for root, dirs, files in os.walk(path):
            for file in files:
                if (
                    file.endswith(".pddl") or file.endswith(".sas")
                ) and not file.startswith("domain"):
                    instances.append(os.path.join(root,file))
        self.config["instance_set"] = instances

        if instances[0].endswith(".pddl"):
            self.config.domain_file = self.config.instance_set_path + "/domain.pddl"
