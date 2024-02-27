"""Wrapper for casting MultiDiscrete action spaces to Discrete."""
from __future__ import annotations

import itertools

import numpy as np
from gymnasium import Wrapper, spaces


class MultiDiscreteActionWrapper(Wrapper):
    """Wrapper to cast MultiDiscrete action spaces to Discrete.
    This should improve usability with standard RL libraries.
    """

    def __init__(self, env):
        """Initialize wrapper.

        Parameters
        ----------
        env : gym.Env
            Environment to wrap

        """
        super().__init__(env)
        self.n_actions = len(self.env.action_space.nvec)
        self.action_space = spaces.Discrete(np.prod(self.env.action_space.nvec))
        self.action_mapper = {}
        for idx, prod_idx in zip(
            range(np.prod(self.env.action_space.nvec)),
            itertools.product(*[np.arange(val) for val in self.env.action_space.nvec]),
            strict=False,
        ):
            self.action_mapper[idx] = prod_idx

    def step(self, action):
        """Maps discrete action value to array."""
        action = self.action_mapper[action]
        return self.env.step(action)
