from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs.onell_env import OneLLEnv

import os
import pandas as pd


class OneLLBenchmark(AbstractBenchmark):
    """
    Benchmark with various settings for (1+(lbd, lbd))-GA
    """

    def __init__(
        self,
        config_path=os.path.dirname(os.path.abspath(__file__))
        + "/../additional_configs/onell/lbd_theory.json",
        config=None
    ):
        """
        Initialize OneLL benchmark

        Parameters
        -------
        config_name: str
            OneLL's config name
            possible values: 'lbd_a', 'lbd_b', 'lbd_p_c', 'lbd1_lbd2_p_c'
        config_path : str
            path to config file (optional)
            all options specified in config_path will override the ones in config_name

        """
        if config_path is None and config is None:
            config_path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/../additional_configs/onell/lbd_theory.json"
            )
        super(OneLLBenchmark, self).__init__(config_path, config)

        self.read_instance_set()

    def get_environment(self):
        """
        Return an environment with current configuration
        """

        env = OneLLEnv(self.config)
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self):
        """Read instance set from file"""
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/"
            + self.config.instance_set_path
        )
        self.config["instance_set"] = pd.read_csv(path, index_col=0).to_dict("id")

        for key, val in self.config["instance_set"].items():
            self.config["instance_set"][key] = objdict(val)
