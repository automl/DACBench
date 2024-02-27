"""Wrapper for reward noise."""
from __future__ import annotations

import numpy as np
from gymnasium import Wrapper


class RewardNoiseWrapper(Wrapper):
    """Wrapper to add noise to the reward signal.

    Noise can be sampled from a custom distribution
    or any distribution in numpy's random module.
    """

    def __init__(
        self, env, noise_function=None, noise_dist="standard_normal", dist_args=None
    ):
        """Initialize wrapper.

        Either noise_function or noise_dist and dist_args need to be given

        Parameters
        ----------
        env : gym.Env
            Environment to wrap
        noise_function : function
            Function to sample noise from
        noise_dist : str
            Name of distribution to sample noise from
        dist_args : list
            Arguments for noise distribution

        """
        super().__init__(env)

        if noise_function:
            self.noise_function = noise_function
        elif noise_dist:
            self.noise_function = self.add_noise(noise_dist, dist_args)
        else:
            raise Exception("No distribution to sample noise from given")

    def __setattr__(self, name, value):
        """Set attribute in wrapper if available and in env if not.

        Parameters
        ----------
        name : str
            Attribute to set
        value
            Value to set attribute to

        """
        if name in ["noise_function", "env", "add_noise", "step"]:
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
        if name in ["noise_function", "env", "add_noise", "step"]:
            return object.__getattribute__(self, name)

        return getattr(self.env, name)

    def step(self, action):
        """Execute environment step and add noise.

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
        reward += self.noise_function()
        reward = max(self.env.reward_range[0], min(self.env.reward_range[1], reward))
        return state, reward, terminated, truncated, info

    def add_noise(self, dist, args):
        """Make noise function from distribution name and arguments.

        Parameters
        ----------
        dist : str
            Name of distribution
        args : list
            List of distribution arguments

        Returns:
        -------
        function
            Noise sampling function

        """
        rng = np.random.default_rng()
        function = getattr(rng, dist)

        def sample_noise():
            if args:
                return function(*args)

            return function()

        return sample_noise
