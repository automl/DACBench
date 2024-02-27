"""Dynamic Random Agent."""
from __future__ import annotations

from gymnasium import spaces

from dacbench.abstract_agent import AbstractDACBenchAgent


class DynamicRandomAgent(AbstractDACBenchAgent):
    """Dynamic random agent class."""

    def __init__(self, env, switching_interval):
        """Initialise the dynamic random agent."""
        self.sample_action = env.action_space.sample
        self.switching_interval = switching_interval
        self.count = 0
        self.action = self.sample_action()
        self.shortbox = (
            isinstance(env.action_space, spaces.Box) and len(env.action_space.low) == 1
        )

    def act(self, state, reward):
        """Return action."""
        if self.count >= self.switching_interval:
            self.action = self.sample_action()
            self.count = 0
        self.count += 1

        if self.shortbox:
            return self.action[0]
        return self.action

    def train(self, next_state, reward):
        """Train agent."""
        pass  # noqa: PIE790

    def end_episode(self, state, reward):
        """End episode."""
        pass  # noqa: PIE790
