"""Policy for sgd ca."""
from __future__ import annotations

import math

from dacbench.abstract_agent import AbstractDACBenchAgent


class CosineAnnealingAgent(AbstractDACBenchAgent):
    """Agent using cosine annea."""

    def __init__(self, env, base_lr=0.1, t_max=1000, eta_min=0):
        """Initialize the Agent."""
        self.eta_min = eta_min
        self.t_max = t_max
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.last_epoch = -1
        super().__init__(env)

    def act(self, state=None, reward=None):
        """Returns the next action."""
        self.last_epoch += 1
        if self.last_epoch == 0:
            return self.base_lr
        if (self.last_epoch - 1 - self.t_max) % (2 * self.t_max) == 0:
            return (
                self.current_lr
                + (self.base_lr - self.eta_min)
                * (1 - math.cos(math.pi / self.t_max))
                / 2
            )
        return (1 + math.cos(math.pi * self.last_epoch / self.t_max)) / (
            1 + math.cos(math.pi * (self.last_epoch - 1) / self.t_max)
        ) * (self.current_lr - self.eta_min) + self.eta_min

    def train(self, state=None, reward=None):  # noqa: D102
        pass

    def end_episode(self, state=None, reward=None):  # noqa: D102
        pass
