import itertools
import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from modcma import Parameters

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import CMAStepSizeEnv, ModCMAEnv

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
ACTIVE = CSH.CategoricalHyperparameter(name="0_active", choices=[True, False])
ELITIST = CSH.CategoricalHyperparameter(name="1_elitist", choices=[True, False])
ORTHOGONAL = CSH.CategoricalHyperparameter(name="2_orthogonal", choices=[True, False])
SEQUENTIAL = CSH.CategoricalHyperparameter(name="3_sequential", choices=[True, False])
THRESHOLD_CONVERGENCE = CSH.CategoricalHyperparameter(
    name="4_threshold_convergence", choices=[True, False]
)
STEP_SIZE_ADAPTION = CSH.CategoricalHyperparameter(
    name="5_step_size_adaption",
    choices=["csa", "tpa", "msr", "xnes", "m-xnes", "lp-xnes", "psr"],
)
MIRRORED = CSH.CategoricalHyperparameter(
    name="6_mirrored", choices=["None", "mirrored", "mirrored pairwise"]
)
BASE_SAMPLER = CSH.CategoricalHyperparameter(
    name="7_base_sampler", choices=["gaussian", "sobol", "halton"]
)
WEIGHTS_OPTION = CSH.CategoricalHyperparameter(
    name="8_weights_option", choices=["default", "equal", "1/2^lambda"]
)
LOCAL_RESTART = CSH.CategoricalHyperparameter(
    name="90_local_restart", choices=["None", "IPOP", "BIPOP"]
)
BOUND_CORRECTION = CSH.CategoricalHyperparameter(
    name="91_bound_correction",
    choices=["None", "saturate", "unif_resample", "COTN", "toroidal", "mirror"],
)

DEFAULT_CFG_SPACE.add_hyperparameter(ACTIVE)
DEFAULT_CFG_SPACE.add_hyperparameter(ELITIST)
DEFAULT_CFG_SPACE.add_hyperparameter(ORTHOGONAL)
DEFAULT_CFG_SPACE.add_hyperparameter(SEQUENTIAL)
DEFAULT_CFG_SPACE.add_hyperparameter(THRESHOLD_CONVERGENCE)
DEFAULT_CFG_SPACE.add_hyperparameter(STEP_SIZE_ADAPTION)
DEFAULT_CFG_SPACE.add_hyperparameter(MIRRORED)
DEFAULT_CFG_SPACE.add_hyperparameter(BASE_SAMPLER)
DEFAULT_CFG_SPACE.add_hyperparameter(WEIGHTS_OPTION)
DEFAULT_CFG_SPACE.add_hyperparameter(LOCAL_RESTART)
DEFAULT_CFG_SPACE.add_hyperparameter(BOUND_CORRECTION)

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
        "config_space": DEFAULT_CFG_SPACE,
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
        "reward_range": (-(10**12), 0),
        "budget": 100,
        "cutoff": 1e6,
        "seed": 0,
        "multi_agent": False,
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
        if (
            "test_set" not in self.config.keys()
            and "test_set_path" in self.config.keys()
        ):
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
