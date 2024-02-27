"""Generic Agent."""

from __future__ import annotations

from dacbench.abstract_agent import AbstractDACBenchAgent


class GenericAgent(AbstractDACBenchAgent):
    """Generic Agent class."""

    def __init__(self, env, policy):
        """Initialize the generic agent."""
        self.policy = policy
        self.env = env

    def act(self, state, reward):
        """Returns action."""
        return self.policy(self.env, state)

    def train(self, next_state, reward):
        """Train agent."""

    def end_episode(self, state, reward):
        """End episode."""
