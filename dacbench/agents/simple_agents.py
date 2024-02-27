"""Simple Agents."""

from __future__ import annotations

from gymnasium import spaces

from dacbench.abstract_agent import AbstractDACBenchAgent


class RandomAgent(AbstractDACBenchAgent):
    """Random Agent class."""

    def __init__(self, env):
        """Initialize the random agent."""
        self.sample_action = env.action_space.sample
        self.shortbox = isinstance(env.action_space, spaces.Box)
        if self.shortbox:
            self.shortbox = self.shortbox and len(env.action_space.low) == 1

    def act(self, state, reward):
        """Returns action."""
        if self.shortbox:
            return self.sample_action()[0]
        return self.sample_action()

    def train(self, next_state, reward):
        """Train agent."""
        pass  # noqa: PIE790

    def end_episode(self, state, reward):
        """End Episode."""
        pass  # noqa: PIE790


class StaticAgent(AbstractDACBenchAgent):
    """Static Agent class."""

    def __init__(self, env, action):
        """Initialize the static agent."""
        self.action = action

    def act(self, state, reward):
        """Returns action."""
        return self.action

    def train(self, next_state, reward):
        """Train Agent."""
        pass  # noqa: PIE790

    def end_episode(self, state, reward):
        """End Episode."""
        pass  # noqa: PIE790
