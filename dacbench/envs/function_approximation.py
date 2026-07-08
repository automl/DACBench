"""Function Approximation Environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dacbench import AbstractMADACEnv

if TYPE_CHECKING:
    from dacbench.envs.env_utils.toy_functions import AbstractFunction


@dataclass
class FunctionApproximationInstance:
    """Function Approximation Instance."""

    functions: list[AbstractFunction]
    dimension_importances: list[float]
    discrete: list[bool]
    omit_instance_type: bool = False


class FunctionApproximationEnv(AbstractMADACEnv):
    """Function Approximation Environment."""

    def __init__(self, config):
        """Initialize Function Approximation Environment."""
        super().__init__(config)
        self.done = False
        self.get_reward = config.get("reward_function", self.get_default_reward)
        self.get_state = config.get("state_method", self.get_default_state)

    def reset(self, seed=None, options=None):
        """Reset environment."""
        if options is None:
            options = {}
        super().reset_(seed)
        self.functions = self.instance.functions
        self.discrete = self.instance.discrete
        self.omit_instance_type = self.instance.omit_instance_type
        self.last_action = None
        return self.get_state(self), {}

    def step(self, action):
        """Step function: check how close the action is to the target."""
        self.done = super().step_()
        # apply action per dimension
        self.distances = []
        action_items = action.values() if isinstance(action, dict) else  np.atleast_1d(action)
        for i, a in enumerate(action_items):
            target = self.functions[i](self.n_steps)
            value = a
            if self.discrete[i]:
                value = np.linspace(0, 1, self.discrete[i])[int(a)]
            dim_distance = np.abs(target - value)
            if isinstance(dim_distance, np.ndarray):
                dim_distance = dim_distance.item()
            self.distances.append(dim_distance)
        self.last_action = list(action_items)
        for i in range(len(self.last_action)):
            if isinstance(self.last_action[i], list | np.ndarray):
                self.last_action[i] = self.last_action[i][0]
        self.weighted_distances = np.array(self.distances) * np.array(
            self.instance.dimension_importances
        )
        return (
            self.get_state(self),
            self.get_reward(self),
            False,
            self.done,
            {
                "raw_distances": self.distances,
                "weighted_distances": self.weighted_distances,
            },
        )

    def get_default_reward(self, _):
        """Get default reward: muliply dimensions."""
        r = np.prod(self.weighted_distances)
        return max(self.reward_range[0], min(self.reward_range[1], r))

    def get_sum_reward(self, _):
        """Get sum reward."""
        r = -np.sum(self.weighted_distances)
        return max(self.reward_range[0], min(self.reward_range[1], r))

    def get_default_state(self, _):
        """Get default state representation."""
        remaining_budget = self.n_steps - self.c_step
        next_state = [int(remaining_budget)]
        for f in self.functions:
            instance_description = f.instance_description
            if self.omit_instance_type:
                instance_description = instance_description[1:]
            next_state += instance_description
        if self.c_step == 0:
            next_state += [-1 for _ in range(len(self.functions))]
        else:
            next_state = np.array(list(next_state) + list(self.last_action))
        return np.array(next_state).astype(float)

    def close(self):
        """Close environment."""
        del self.instance_set
        return True
