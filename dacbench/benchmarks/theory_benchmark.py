"""Theory Benchmark."""

from __future__ import annotations

from pathlib import Path

import ConfigSpace as CS  # noqa: N817
import ConfigSpace.hyperparameters as CSH
import gymnasium as gym
import numpy as np
import pandas as pd

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs.theory import TheoryEnv, TheoryEnvDiscrete

INFO = {
    "identifier": "Theory",
    "name": "DAC benchmark with RLS algorithm and LeadingOne problem",
    "reward": "Negative number of iterations until solution",
    "state_description": "specified by user",
}

THEORY_DEFAULTS = {
    "observation_description": "n, f(x)",  # examples: n, f(x), delta_f(x), optimal_k,
    # k, k_{t-0..4}, f(x)_{t-1}, f(x)_{t-0..4}
    "reward_range": [-np.inf, np.inf],  # the true reward range is instance dependent
    "reward_choice": "imp_minus_evals",  # see envs/theory.py for more details
    "cutoff": 1e6,  # if using as a "train" environment, a cutoff of 0.8*n^2 where n
    # is problem size will be used (for more details, please see https://arxiv.org/abs/2202.03259)
    # see get_environment function of TheoryBenchmark on how to specify
    # a train/test environment
    "seed": 0,
    "seed_action_space": False,  # set this one to True for reproducibility when random
    # action is sampled in the action space with gym.action_space.sample()
    "problem": "LeadingOne",  # possible values: "LeadingOne"
    "instance_set_path": "lo_rls_50.csv",  # if the instance list file cannot be found
    # in the running directory, it will be
    # looked up in
    # <DACBench>/dacbench/instance_sets/theory/
    "discrete_action": True,  # action space is discrete
    "action_choices": [1, 2, 4, 8, 16],  # portfolio of k values
    "benchmark_info": INFO,
    "name": "LeadingOnesDAC",
}


class TheoryBenchmark(AbstractBenchmark):
    """Benchmark with various settings for (1+(lbd, lbd))-GA and RLS."""

    def __init__(self, config=None):
        """Initialize a theory benchmark.

        Parameters
        -------
        base_config_name: str
            OneLL's config name
            possible values: see ../additional_configs/onell/configs.py
        config : str
            a dictionary, all options specified in this argument will override the one
            in base_config_name

        """
        super().__init__()

        self.config = objdict(THEORY_DEFAULTS)

        if config:
            for key, val in config.items():
                self.config[key] = val

        self.read_instance_set()

        # initialise action space and environment class
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
            assert (
                "max_action" not in self.config
            ), "ERROR: max_action should not be used for discrete action space"
            self.config.env_class = "TheoryEnvDiscrete"
            n_acts = len(self.config["action_choices"])
            action = CSH.UniformIntegerHyperparameter(
                name="", lower=0, upper=n_acts - 1
            )
        else:
            if "action_choices" not in self.config:
                raise ValueError(
                    "WARNING: 'action_choices' is ignored in continuous action spaces."
                )
            assert ("min_action" in self.config) and (  # noqa: PT018
                "max_action" in self.config
            ), "ERROR: min_action and max_action must be specified"
            self.config.env_class = "TheoryEnv"
            action = CSH.UniformFloatHyperparameter(
                name="Step_size",
                lower=self.config["min_action"],
                upper=self.config["max_action"],
            )

        cfg_space.add(action)
        self.config["config_space"] = cfg_space

        # create observation space
        self.env_class = globals()[self.config.env_class]
        assert self.env_class in (TheoryEnv, TheoryEnvDiscrete)

        self.config[
            "observation_space"
        ] = self.create_observation_space_from_description(
            self.config["observation_description"], self.env_class
        )

    def create_observation_space_from_description(
        self, obs_description, env_class=TheoryEnvDiscrete
    ):
        """Create a gym observation space (Box only) based on a string containing
        observation variable names, e.g. "n, f(x), k, k_{t-1}".

        Return:
            A gym.spaces.Box observation space.
        """
        obs_var_names = [s.strip() for s in obs_description.split(",")]
        low = []
        high = []
        for var_name in obs_var_names:
            l, h = env_class.get_obs_domain_from_name(var_name=var_name)  # noqa: E741
            low.append(l)
            high.append(h)
        return gym.spaces.Box(low=np.array(low), high=np.array(high))

    def get_environment(self, test_env=False):
        """Return an environment with current configuration.

        Parameters:
            test_env:   whether the enviroment is used for train an agent or for testing
                        if test_env=False:
                            cutoff time for an episode is set to 0.8*n^2
                            (n: problem size)
                            if an action is out of range, stop the episode immediately
                            and return a large negative reward (see envs/theory.py for
                            more details)
                        otherwise: benchmark's original cutoff time is used,
                            and out-of-range action will be clipped to nearest valid
                            value and the episode will continue.
        """
        env = self.env_class(self.config, test_env)

        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self):
        """Read instance set from file
        we look at the current directory first,
        if the file doesn't exist, we look in <DACBench>/dacbench/instance_sets/theory/.
        """
        assert self.config.instance_set_path
        if Path(self.config.instance_set_path).is_file():
            path = self.config.instance_set_path
        else:
            path = (
                Path(__file__).resolve().parent
                / "../instance_sets/theory/"
                / self.config.instance_set_path
            )

        instance_df = pd.read_csv(path, index_col=0)
        self.config["instance_set"] = instance_df.to_dict(orient="index")

        assert len(self.config["instance_set"].items()) > 0, "ERROR: empty instance set"
        assert (
            "initObj" in self.config["instance_set"][0]
        ), "ERROR: initial solution (initObj) must be specified in instance set"
        assert (
            "size" in self.config["instance_set"][0]
        ), "ERROR: problem size must be specified in instance set"

        for key, val in self.config["instance_set"].items():
            self.config["instance_set"][key] = objdict(val)
