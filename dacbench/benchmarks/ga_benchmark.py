"""GA Benchmark.

Dynamic control of the mutation strength r of the (1+1) EA or the population size
lambda of the (1+lambda) EA and the (1+(lambda,lambda)) GA on OneMax and
LeadingOnes (see dacbench/envs/ga.py).

Prior art: the OneMax DAC extension (https://github.com/taindp98/OneMax-DAC-Extension,
arXiv:2502.20265, arXiv:2505.12982, arXiv:2606.10129) controls lambda of the
(1+(lambda,lambda)) GA on OneMax with n in {50, 100, 200, 300, 500, 1000, 2000},
starting from a solution with n/2 one-bits, observation "n, f(x)", reward
"imp_minus_evals" and the portfolio {1, 2, 4, ..., 2^floor(log2 n)} (e.g.
[1, ..., 32] for n=50, [1, ..., 256] for n=500). The instance sets
onemax_{300,500,1000,2000}.csv mirror these sizes. The default portfolio here is
[1, 2, 4, 8, 16], kept from the Theory benchmark for comparability; pass
action_choices to reproduce the prior-art portfolio.
"""

from __future__ import annotations

from pathlib import Path

import ConfigSpace as CS  # noqa: N817
import ConfigSpace.hyperparameters as CSH
import gymnasium as gym
import numpy as np
import pandas as pd

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs.ga import (
    HISTORY_LENGTH,
    HISTORY_NAMES,
    OBS_DOMAINS,
    GAEnv,
)

#: Observation variables and their history variants, derived from ``envs/ga.py``
#: so that the benchmark info cannot drift from the environment.
_OBS_VARIABLES = ", ".join(OBS_DOMAINS)
_HISTORY_VARIABLES = ", ".join(f"{name}_{{t-k}}" for name in HISTORY_NAMES)

INFO = {
    "identifier": "GA",
    "name": (
        "DAC benchmark controlling the (1+1) EA, (1+lambda) EA or "
        "(1+(lambda,lambda)) GA on OneMax or LeadingOnes"
    ),
    "reward": (
        "Fitness improvement minus number of evaluations per generation by default "
        "(see reward_choice)"
    ),
    "state_description": (
        "Specified by observation_description (default: n, f(x)); available: "
        f"{_OBS_VARIABLES}, {_HISTORY_VARIABLES} "
        f"for k in 0..{HISTORY_LENGTH - 1}"
    ),
}

GA_DEFAULTS = objdict(
    {
        "algorithm": "one_plus_one",  # one_plus_one, one_plus_lambda,
        # one_plus_lambda_lambda
        "problem": "OneMax",  # OneMax or LeadingOne
        "discrete_action": True,  # continuous actions only for one_plus_one
        "action_choices": [1, 2, 4, 8, 16],  # portfolio of r or lambda values
        "observation_description": "n, f(x)",
        "reward_choice": "imp_minus_evals",  # see envs/ga.py for more details
        "reward_range": [-np.inf, np.inf],  # the true reward range is instance
        # dependent
        "cutoff": 1e6,  # if used as a "train" environment, a cutoff of 0.8*n^2
        # evaluations is used instead (as in the Theory benchmark)
        "seed": 0,
        "seed_action_space": False,
        "count_different_inds_only": True,  # practice-aware evaluation counting
        # for one_plus_lambda_lambda
        "instance_set_path": "onemax_50.csv",  # looked up in the running directory
        # first, then in <DACBench>/dacbench/instance_sets/ga/
        "benchmark_info": INFO,
        "name": "GA",
    }
)


class GABenchmark(AbstractBenchmark):
    """Benchmark for (1+1) EA, (1+lambda) EA and (1+(lambda,lambda)) GA."""

    @staticmethod
    def _isolate_defaults(config):
        """Copy the mutable defaults so that per-instance mutations do not leak
        into the module-level ``INFO`` dict and ``GA_DEFAULTS``.  Idempotent:
        safe to call multiple times.
        """
        config["benchmark_info"] = objdict(config["benchmark_info"])
        config["action_choices"] = list(config["action_choices"])

    def __init__(self, config=None):
        """Initialize a GA benchmark with ``GA_DEFAULTS`` overridden by ``config``."""
        super().__init__()

        self.config = objdict(GA_DEFAULTS.copy())

        if config:
            for key, val in config.items():
                self.config[key] = val

        self._isolate_defaults(self.config)

        self.read_instance_set()

        # initialise action space
        cfg_space = CS.ConfigurationSpace()
        if self.config.discrete_action:
            assert (
                "action_choices" in self.config
            ), "ERROR: action_choices must be specified"
            assert ("min_action" not in self.config) and (  # noqa: PT018
                "max_action" not in self.config
            ), (
                "ERROR: min_action and max_action should not be used for "
                "discrete action space"
            )
            n_acts = len(self.config["action_choices"])
            if n_acts == 1:
                # ConfigSpace rejects integer ranges with lower == upper
                action = CSH.CategoricalHyperparameter(name="", choices=[0])
            else:
                action = CSH.UniformIntegerHyperparameter(
                    name="", lower=0, upper=n_acts - 1
                )
        else:
            if self.config.algorithm != "one_plus_one":
                raise ValueError(
                    f"ERROR: algorithm {self.config.algorithm} controls the "
                    "population size lambda and requires a discrete action space "
                    "(discrete_action=True with action_choices)."
                )
            if "min_action" not in self.config:
                self.config["min_action"] = 1
            if "max_action" not in self.config:
                self.config["max_action"] = max(
                    int(inst.size) for inst in self.config["instance_set"].values()
                )
            action = CSH.UniformFloatHyperparameter(
                name="r",
                lower=self.config["min_action"],
                upper=self.config["max_action"],
            )

        cfg_space.add(action)
        self.config["config_space"] = cfg_space

        # create observation space
        self.env_class = GAEnv
        self.config[
            "observation_space"
        ] = self.create_observation_space_from_description(
            self.config["observation_description"], self.env_class
        )

    def create_observation_space_from_description(
        self, obs_description, env_class=GAEnv
    ):
        """Create a Box observation space from a comma-separated variable list."""
        obs_var_names = [s.strip() for s in obs_description.split(",")]
        low = []
        high = []
        for var_name in obs_var_names:
            lower, upper = env_class.get_obs_domain_from_name(var_name=var_name)
            low.append(lower)
            high.append(upper)
        return gym.spaces.Box(low=np.array(low), high=np.array(high))

    def get_environment(self, test_env=False):
        """Return an environment with current configuration.

        Parameters
        ----------
        test_env : bool
            whether the environment is used for training an agent or for testing.
            If False, the evaluation budget of an episode is 0.8*n^2 (n: problem
            size), and an out-of-range action stops the episode immediately with
            a large negative reward (see envs/ga.py for more details). Otherwise,
            the benchmark's cutoff is used as evaluation budget, and out-of-range
            actions are clipped to the nearest valid value and the episode
            continues.

        """
        env = self.env_class(self.config, test_env)

        for func in self.wrap_funcs:
            env = func(env)

        return env

    @staticmethod
    def _read_instance_file(instance_set_path):
        """Read an instance file, looking in the current directory first and then
        in <DACBench>/dacbench/instance_sets/ga/.
        """
        if Path(instance_set_path).is_file():
            path = instance_set_path
        else:
            path = (
                Path(__file__).resolve().parent
                / "../instance_sets/ga/"
                / instance_set_path
            )

        instance_df = pd.read_csv(path, index_col=0)
        instance_set = instance_df.to_dict(orient="index")

        assert len(instance_set.items()) > 0, "ERROR: empty instance set"
        first = next(iter(instance_set.values()))
        assert (
            "initObj" in first
        ), "ERROR: initial solution (initObj) must be specified in instance set"
        assert "size" in first, "ERROR: problem size must be specified in instance set"

        return {key: objdict(val) for key, val in instance_set.items()}

    def read_instance_set(self):
        """Read instance set (and test set, if test_set_path is given) from file
        we look at the current directory first,
        if the file doesn't exist, we look in <DACBench>/dacbench/instance_sets/ga/.
        """
        assert self.config.instance_set_path
        self.config["instance_set"] = self._read_instance_file(
            self.config.instance_set_path
        )
        if self.config.get("test_set_path"):
            self.config["test_set"] = self._read_instance_file(
                self.config.test_set_path
            )
