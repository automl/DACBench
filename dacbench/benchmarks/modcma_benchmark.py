import os
import itertools

import numpy as np
from modcma import Parameters

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import ModCMAEnv, CMAStepSizeEnv


INFO = {
    "identifier": "ModCMA",
    "name": "Online Selection of CMA-ES Variants",
    "reward": "Negative best function value",
    "state_description": [
        "Generation Size",
        "Sigma",
        "Remaining Budget",
        "Function ID",
        "Instance ID",
    ],
}


MODCMA_DEFAULTS = objdict(
    {
        "action_space_class": "MultiDiscrete",
        "action_space_args": [
            list(
                map(
                    lambda m: len(
                        getattr(getattr(Parameters, m), "options", [False, True])
                    ),
                    Parameters.__modules__,
                )
            )
        ],
        "observation_space_class": "Box",
        "observation_space_args": [-np.inf * np.ones(5), np.inf * np.ones(5)],
        "observation_space_type": np.float32,
        "reward_range": (-(10 ** 12), 0),
        "budget": 100,
        "cutoff": 1e6,
        "seed": 0,
        "instance_set_path": os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../instance_sets/modea/modea_train.csv",
        ),
        "test_set_path": os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../instance_sets/modea/modea_train.csv",
        ),
        "benchmark_info": INFO,
    }
)


class ModCMABenchmark(AbstractBenchmark):
    def __init__(self, config_path: str = None, step_size=False, config=None):
        super().__init__(config_path, config)
        self.config = objdict(MODCMA_DEFAULTS.copy(), **(self.config or dict()))
        self.step_size = step_size

    def get_environment(self):
        if "instance_set" not in self.config:
            self.read_instance_set()

        # Read test set if path is specified
        if "test_set" not in self.config.keys() and "test_set_path" in self.config.keys():
            self.read_instance_set(test=True)

        if self.step_size:
            self.config.action_space_class = "Box"
            self.config.action_space_args = [np.array([0]), np.array([10])]
            env = CMAStepSizeEnv(self.config)
        else:
            env = ModCMAEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)
        return env

    def read_instance_set(self, test=False):
        if test:
            path = self.config.test_set_path
            keyword = "test_set"
        else:
            path = self.config.instance_set_path
            keyword = "instance_set"

        self.config[keyword] = dict()
        with open(path, "r") as fh:
            for line in itertools.islice(fh, 1, None):
                _id, dim, fid, iid, *representation = line.strip().split(",")
                self.config[keyword][int(_id)] = [
                    int(dim),
                    int(fid),
                    int(iid),
                    list(map(int, representation)),
                ]

    def get_benchmark(self, seed: int = 0):
        self.config = MODCMA_DEFAULTS.copy()
        self.config.seed = seed
        self.read_instance_set()
        self.read_instance_set(test=True)
        return ModCMAEnv(self.config)
