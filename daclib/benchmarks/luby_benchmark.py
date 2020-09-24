import AbstractBenchmark
from gym import spaces
import numpy as np

MAX_STEPS = 2**6
LUBY_SEQUENCE = np.log2([next(luby_gen(i)) for i in range(1, 2*MAX_STEPS + 2)])
HISTORY_LENGTH = 5

LUBY_DEFAULTS = {
    "action_space": "Discrete",
    "action_space_args": [int(np.log2(MAX_STEPS))],
    "observation_space": "Box",
    "observation_space_type": np.float32,
    "observation_space_args": [np.array([-1 for _ in range(self._hist_len + 1)]), np.array([2**max(LUBY_SEQUENCE + 1) for _ in range(HISTORY_LENGTH + 1)])],
    "reward_range": (-1, 0),
    "cutoff" : MAX_STEPS,
    "hist_lenght": HISTORY_LENGTH,
    "min_steps": 2**3,
    "fuzzy": False
}

class LubyBenchmark(AbstractBenchmark):
    def __init__(self, config_path):
        super(LubyBenchmark, self).__init__(config_path)
        if not self.config:
            self.config = LUBY_DEFAULTS

        for key in LUBY_DEFAULTS:
            if not key in self.config:
                self.config[key] = LUBY_DEFAULTS[key]

    def get_benchmark_env(self):
        return LubyEnv(config)
