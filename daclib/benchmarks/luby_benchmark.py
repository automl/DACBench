from daclib.abstract_benchmark import AbstractBenchmark, objdict
from daclib.envs import LubyEnv, luby_gen

from gym import spaces
import numpy as np

MAX_STEPS = 2 ** 6
LUBY_SEQUENCE = np.log2([next(luby_gen(i)) for i in range(1, 2 * MAX_STEPS + 2)])
HISTORY_LENGTH = 5

LUBY_DEFAULTS = objdict(
    {
        "action_space": "Discrete",
        "action_space_args": [int(np.log2(MAX_STEPS))],
        "observation_space": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [
            np.array([-1 for _ in range(HISTORY_LENGTH + 1)]),
            np.array([2 ** max(LUBY_SEQUENCE + 1) for _ in range(HISTORY_LENGTH + 1)]),
        ],
        "reward_range": (-1, 0),
        "cutoff": MAX_STEPS,
        "hist_length": HISTORY_LENGTH,
        "min_steps": 2 ** 3,
        "fuzzy": False,
        "instance_set": "../instance_sets/luby_train.csv",
    }
)


class LubyBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for Sigmoid
    """

    def __init__(self, config_path=None):
        super(LubyBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = LUBY_DEFAULTS

        for key in LUBY_DEFAULTS:
            if not key in self.config:
                self.config[key] = LUBY_DEFAULTS[key]

    def get_benchmark_env(self):
        """
        Return Luby env with current configuration

        Returns
        -------
        LubyEnv
            Luby environment
        """
        return LubyEnv(self.config)

    def set_cutoff(self, steps):
        """
        Set cutoff and adapt dependencies

        Parameters
        -------
        int
            Maximum number of steps
        """
        self.config.cutoff = steps
        self.config.action_space_args = [int(np.log2(steps))]
        luby_seq = np.log2([next(luby_gen(i)) for i in range(1, 2 * steps+ 2)])
        self.config.observation_space_args= [np.array([-1 for _ in range(self.config.hist_length + 1)]), np.array([2 ** max(luby_seq + 1) for _ in range(self.config.hist_length+ 1)])]

    def set_history_length(self, length):
        """
        Set history length and adapt dependencies

        Parameters
        -------
        int
            History length
        """
        self.config.hist_length = length
        self.config.observation_space_args= [np.array([-1 for _ in range(length + 1)]), np.array([2 ** max(LUBY_SEQUENCE + 1) for _ in range(length+ 1)])]
