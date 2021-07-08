"""
Geometric environment.
Original environment authors: Rasmus von Glahn
"""
import numpy as np
from typing import List

from dacbench import AbstractEnv


class GeometricEnv(AbstractEnv):
    """
    Environment for tracing difernet curves that are orthogonal to each other
    Use product approach: f(x,y,z) = X(x) * Y(y) * Z(z)
    """

    def _sig(self, x, scaling, inflection):
        """ Simple sigmoid function """
        return 1 / (1 + np.exp(-scaling * (x - inflection)))

    def _linear(self, x: float, a: float, b: float):
        """ Linear function """
        return a * x + b

    def _polynom(self, x: float, coeff_list: List[float]):
        """ Polynomial function. Dimension depends on length of coefficient list. """
        pol_value = 0

        for dim, coeff in enumerate(coeff_list):
            pol_value += coeff * x ** dim

        return pol_value

    def _constant(self, c: float):
        return c

    def __init__(self, config) -> None:
        """
        Initialize Geometric Env

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        pass

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
        pass

    def reset(self) -> List[int]:
        """
        Resets env

        Returns
        -------
        numpy.array
            Environment state
        """
        pass

    def get_default_reward(self, _):
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
        pass

    def render(self, mode: str) -> None:
        """
        Render env in human mode

        Parameters
        ----------
        mode : str
            Execution mode
        """
        pass
