"""Wrapper that converts observation spaces to spaces.Box."""
from __future__ import annotations

import numpy as np
from gymnasium import Wrapper, spaces


class ObservationWrapper(Wrapper):
    """Wrapper convert observations spaces to spaces.Box for convenience.

    Currently only supports Dict -> Box
    """

    def __init__(self, env):
        """Initialize wrapper.

        Parameters
        ----------
        env : gym.Env
            Environment to wrap
        compute_optimal : function
            Function to compute optimal policy

        """
        super().__init__(env)
        obs_sample = self.flatten(self.env.observation_space.sample())
        size = len(obs_sample)
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(size), high=np.inf * np.ones(size)
        )

    def __setattr__(self, name, value):
        """Set attribute in wrapper if available and in env if not.

        Parameters
        ----------
        name : str
            Attribute to set
        value
            Value to set attribute to

        """
        if name in ["observation_space", "step", "env", "flatten", "reset"]:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def __getattribute__(self, name):
        """Get attribute value of wrapper if available and of env if not.

        Parameters
        ----------
        name : str
            Attribute to get

        Returns:
        -------
        value
            Value of given name

        """
        if name in ["observation_space", "step", "env", "flatten", "reset"]:
            return object.__getattribute__(self, name)
        return getattr(self.env, name)

    def step(self, action):
        """Execute environment step and record distance.

        Parameters
        ----------
        action : int
            action to execute

        Returns:
        -------
        np.array, float, bool, bool, dict
            state, reward, terminated, truncated, metainfo

        """
        state, reward, terminated, truncated, info = self.env.step(action)
        state = self.flatten(state)
        return state, reward, terminated, truncated, info

    def reset(self):
        """Execute environment step and record distance.

        Returns:
        -------
        np.array, dict
            state, info

        """
        state, info = self.env.reset()
        state = self.flatten(state)
        return state, info

    def flatten(self, state_dict):
        """Flatten dict to list."""
        keys = sorted(state_dict.keys())
        values = []
        for k in keys:
            if isinstance(state_dict[k], np.ndarray):
                for s in state_dict[k]:
                    values.append(s)
            else:
                values.append(state_dict[k])
        return np.array(values).astype(np.float32)
