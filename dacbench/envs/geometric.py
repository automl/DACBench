"""
Geometric environment.
Original environment authors: Rasmus von Glahn
"""
import numpy as np
import itertools
from typing import List

from dacbench import AbstractEnv


class GeometricEnv(AbstractEnv):
    """
    Environment for tracing different curves that are orthogonal to each other
    Use product approach: f(t,x,y,z) = X(t,x) * Y(t,y) * Z(t,z)
    Normalize Function Value on a Scale betwwen 0 and 1
        - min and max value for normalization over all timesteps
    """

    def _sigmoid(self, t: float, scaling: float, inflection: float):
        """ Simple sigmoid function """
        return 1 / (1 + np.exp(-scaling * (t - inflection)))

    def _linear(self, t: float, a: float, b: float):
        """ Linear function """
        return a * t + b

    def _polynom(self, t: float, coeff_list: List[float]):
        """ Polynomial function. Dimension depends on length of coefficient list. """
        pol_value = 0

        for dim, coeff in enumerate(coeff_list):
            pol_value += coeff * t ** dim

        return pol_value

    def _logarithmic():
        """ Logarithmic function """
        pass

    def _exponential():
        """ Exponential function """
        pass

    def _constant(self, c: float):
        """ Constant function """
        return c

    def __init__(self, config) -> None:
        """
        Initialize Geometric Env

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(GeometricEnv, self).__init__(config)

        self.action_vals = config["action_values"]
        self.n_actions = len(self.action_vals)
        self.action_mapper = {}
        self.function_stack = []
        self.function_params = []

        # Map actions to differnet actions to differnet action configurations
        for idx, prod_idx in zip(
            range(np.prod(config["action_values"])),
            itertools.product(*[np.arange(val) for val in config["action_values"]]),
        ):
            self.action_mapper[idx] = prod_idx

        self._prev_state = None
        self.action = None

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

    def step(self, action: int):
        """
        Execute environment step

        Parameters
        ----------
        action : int
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, info
        """
        self.done = super(GeometricEnv, self).step_()

        action = self.action_mapper[action]
        assert self.n_actions == len(
            action
        ), f"action should be of length {self.n_actions}."
        self.action = action

        next_state = self.get_state(self)
        self._prev_state = next_state
        return next_state, self.get_reward(self), self.done, {}

    def reset(self) -> List[int]:
        """
        Resets env

        Returns
        -------
        numpy.array
            Environment state
        """
        super(GeometricEnv, self).reset_()

    def get_default_reward(self, _):
        # TODO: calculate euklidean distance between curve and actions
        pass

    def get_default_state(self, _):
        pass

    def close(self) -> bool:
        """
        Close Env

        Returns
        -------
        bool
            Closing confirmation
        """
        True

    def render(self, mode: str) -> None:
        """
        Render env in human mode

        Parameters
        ----------
        mode : str
            Execution mode
        """
        pass
