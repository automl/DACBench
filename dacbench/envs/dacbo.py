"""DACBO Env."""

from __future__ import annotations

import numpy as np
from dacboenv.dacboenv import DACBOEnv as DEnv
from dacboenv.env.instance import ExternalInstanceSelector

from dacbench.abstract_env import AbstractEnv


class DACBOEnv(AbstractEnv):
    """DACBO env."""

    def __init__(self, config):
        """Init DACBO env."""
        config["instance_selector_class"] = ExternalInstanceSelector
        self._env = DEnv(**config)
        self._env.reset()  # Init spaces (NOT self.reset())
        config["cutoff"] = np.inf
        config["observation_space"] = self._env.observation_space
        config["action_space"] = self._env.action_space
        super().__init__(config)

    def step(self, action):
        """Takes one env step."""
        super().step_()
        state, reward, terminated, truncated, info = self._env.step(action)
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the internal env."""
        if options is None:
            options = {}
        super().reset_(seed, options)  # AbstractEnv picks next instance
        self._env.instance_selector.set_instance(self.instance)
        obs, info = self._env.reset()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        return obs, info
