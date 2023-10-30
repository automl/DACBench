import csv
import os

from ConfigSpace import ConfigurationSpace, Float
import numpy as np

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import AutoRLEnv

DEFAULT_CFG_SPACE = ConfigurationSpace(
    {
        "update_epochs": (1, 25),
        "num_minibatches": (1, 512),
        "minibatch_size": (1, 256),
        "clip_eps": (0.5, 0.999),
        "ent_coef": (0.0, 1.0),
        "vf_coef": (0.0, 1.0),
        "max_grad_norm": (0.0, 1.0),
        "hidden_size": (4, 2048),
    },
    seed=0,
)
LR = Float("lr", (0.00001, 0.01), log=True)
GAE_LAMBDA = Float("gae_lambda", (0.00001, 0.01), log=True)
DEFAULT_CFG_SPACE.add_hyperparameters([LR, GAE_LAMBDA])

DEFAULT_RL_CONFIG = {
    "lr": 2.5e-4,
    "num_envs": 4,
    "num_steps": 128,
    "total_timesteps": 100,
    "update_epochs": 4,
    "num_minibatches": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "activation": "tanh",
    "env_name": "CartPole-v1",
    "num_eval_episodes": 10,
    "env_framework": "gymnax",
}

INFO = {
    "identifier": "AutoRL",
    "name": "Hyperparameter Control for Reinforcement Learning",
    "reward": "Evaluation Reward",
    "state_description": [],
}

AUTORL_DEFAULTS = objdict(
    {
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [np.array([-1]), np.array([1])],
        "reward_range": (-np.inf, np.inf),
        "cutoff": 1000,
        "seed": 0,
        "benchmark_info": INFO,
        "checkpoint": ["policy"],
        "checkpoint_dir": "autorl_checkpoints",
        "instance_set": {0: DEFAULT_RL_CONFIG},
        "track_trajectory": False,
        "grad_obs": False,
        "algorithm": "ppo",
    }
)


class AutoRLBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for Sigmoid
    """

    def __init__(self, config_path=None, config=None):
        """
        Initialize Luby Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super().__init__(config_path, config)
        self.train_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.test_seeds = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        if not self.config:
            self.config = objdict(AUTORL_DEFAULTS.copy())

        for key in AUTORL_DEFAULTS:
            if key not in self.config:
                self.config[key] = AUTORL_DEFAULTS[key]

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

        env = AutoRLEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self, test=False):
        """Read instance set from file"""
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

        self.config[keyword] = {}
        with open(path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self.config[keyword][int(row["ID"])] = [
                    float(shift) for shift in row["start"].split(",")
                ] + [float(slope) for slope in row["sticky"].split(",")]

    def get_benchmark(
        self,
        seed,
        name="everything",
        test=False,
        level="mix",
        dynamic=False,
        get_data=False,
    ):
        naming = {
            "single_env": "single_env",
            "tune_me_up": "single_env",
            "multi_env": "multi_env",
            "on_your_left": "multi_env",
            "cmdp": "cmdp",
            "to_generalize_is_to_think": "cmdp",
            "cash": "cash",
            "cash_is_king": "cash",
            "full": "full",
            "the_whole_enchilada": "full",
        }
        try:
            config_path = (
                os.path.dirname(os.path.abspath(__file__))
                + f"/../additional_configs/autorl/{naming[name]}_level_{level}"
            )
            if dynamic:
                config_path += "_dynamic"
            if test:
                config_path += "_test"
            config_path += ".json"
            self.read_config_file(config_path)
        except KeyError:
            print(
                "Config file not found, available benchmarks are:\ne 'single_env' or 'tune_me_up': Tuning a single algorithm on a single environment\n 'multi_env' or 'on_your_left': Tuning a single algorithm on multiple environments\n 'cmdp' or 'to_generalize_is_to_think': Tuning a single algorithm on a contextual MDP\n 'cash' or 'cash_is_king': Algorithm selection and tuning on a single environment\n 'full' or 'the_whole_enchilada': Algorithm selection and tuning on multiple environments and contexts"
            )
            return

        if (seed not in self.train_seeds and not test) or (
            seed not in self.test_seeds and test
        ):
            new_seed = np.random.choice(self.train_seeds)
            print(
                f"Seed {seed} not in available seeds (train: 0-9, test: 10-19), defaulting to random train seed {new_seed}."
            )
            seed = new_seed
        self.config.seed = seed
        env = AutoRLEnv(self.config)
        dataset = None
        if get_data:
            data_path = (
                os.path.dirname(os.path.abspath(__file__))
                + f"/../datasets/autorl/{naming[name]}_{level}"
            )
            if dynamic:
                data_path += "_dynamic"
            if test:
                data_path += "_test"

            try:
                dataset = AutoRLDataset(data_path)
                return env, dataset
            except FileNotFoundError:
                print(
                    "Dataset file not found, please make sure you downloaded the data first."
                )
                return
        return env
