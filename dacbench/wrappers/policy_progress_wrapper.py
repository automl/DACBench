"""Wrapper for process tracking."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Wrapper


class PolicyProgressWrapper(Wrapper):
    """Wrapper to track progress towards optimal policy.

    Can only be used if a way to obtain the optimal policy
    given an instance can be obtained.
    """

    def __init__(self, env, compute_optimal):
        """Initialize wrapper.

        Parameters
        ----------
        env : gym.Env
            Environment to wrap
        compute_optimal : function
            Function to compute optimal policy

        """
        super().__init__(env)
        self.compute_optimal = compute_optimal
        self.episode = []
        self.policy_progress = []

    def __setattr__(self, name, value):
        """Set attribute in wrapper if available and in env if not.

        Parameters
        ----------
        name : str
            Attribute to set
        value
            Value to set attribute to

        """
        if name in [
            "compute_optimal",
            "env",
            "episode",
            "policy_progress",
            "render_policy_progress",
        ]:
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
        if name in [
            "step",
            "compute_optimal",
            "env",
            "episode",
            "policy_progress",
            "render_policy_progress",
        ]:
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
        self.episode.append(action)
        if terminated or truncated:
            optimal = self.compute_optimal(self.env.instance)
            self.policy_progress.append(
                np.linalg.norm(np.array(optimal) - np.array(self.episode))
            )
            self.episode = []
        return state, reward, terminated, truncated, info

    def render_policy_progress(self):
        """Plot progress."""
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(self.policy_progress)), self.policy_progress)
        plt.title("Policy progress over time")
        plt.xlabel("Episode")
        plt.ylabel("Distance to optimal policy")
        plt.show()
