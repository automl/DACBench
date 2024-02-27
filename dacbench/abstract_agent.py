"""Abstract Agent."""
from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractDACBenchAgent(ABC):
    """Abstract class to implement for use with the runner function."""

    @abstractmethod
    def __init__(self, env):
        """Initialize agent.

        Parameters
        ----------
        env : gym.Env
            Environment to train on

        """

    @abstractmethod
    def act(self, state, reward):
        """Compute and return environment action.

        Parameters
        ----------
        state
            Environment state
        reward
            Environment reward

        Returns:
        -------
        action
            Action to take

        """
        raise NotImplementedError

    @abstractmethod
    def train(self, next_state, reward):
        """Train during episode if needed (pass if not).

        Parameters
        ----------
        next_state
            Environment state after step
        reward
            Environment reward

        """
        raise NotImplementedError

    @abstractmethod
    def end_episode(self, state, reward):
        """End of episode training if needed (pass if not).

        Parameters
        ----------
        state
            Environment state
        reward
            Environment reward

        """
        raise NotImplementedError
