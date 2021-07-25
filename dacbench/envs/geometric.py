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
        self.action_interval_mapping = config["action_interval_mapping"]
        self.max_function_value = config["max_function_value"]
        self.n_actions = len(self.action_vals)
        self.action_mapper = {}

        # state variables
        self.trajectory_set = {}
        self.derivative_set = {}
        self.trajectory = []
        self.derivative = []
        self.derivative_interval = []

        # map actions from int to vector representation
        for idx, prod_idx in zip(
            range(np.prod(config["action_values"])),
            itertools.product(*[np.arange(val) for val in config["action_values"]]),
        ):
            self.action_mapper[idx] = prod_idx

        self._prev_state = None
        self.action = None
        self._calculate_norm_value()

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

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

    def _logarithmic(self, t: float, a: int):
        """ Logarithmic function """
        if t != 0:
            return a * np.log(t)
        else:
            return self.max_function_value

    def _exponential(self, t: float, a: int):
        """ Exponential function """
        return a * np.exp(t)

    def _constant(self, c: float):
        """ Constant function """
        return c

    def _calculate_norm_value(self):
        """
        Norm Functions to Intervall between -1 and 1
        """
        for key, instance in self.instance_set.items():

            for dim, function_info in enumerate(instance):
                value_list = []

                for step in range(self.n_steps):
                    value_list.append(
                        self._calculate_function_value(step, function_info, True)
                    )

                # set first item of every function_list in instance as norm factor
                if abs(min(value_list)) > max(value_list):
                    norm_factor = abs(min(value_list))
                else:
                    norm_factor = max(value_list)

                self.instance_set[key][dim][0] = norm_factor

    def _calculate_function_value(
        self, time_step: int, function_infos: List, calculate_norm=False
    ) -> float:
        """
        Call differnet functions with their speicifc parameters.

        Parameters
        ----------
        function_infos : List
            Consists of function name and the coefficients
        time_step: int
            time step for each function
        calculate_norm : bool, optional
            True if norm gets calculated, by default False

        Returns
        -------
        float
            coordinate in dimension of function
        """
        norm_value = function_infos[0] if not calculate_norm else 1
        function_name = function_infos[1]
        coefficients = function_infos[2:]

        function_value = 0

        if "sigmoid" == function_name:
            function_value = self._sigmoid(time_step, coefficients[0], coefficients[1])

        elif "linear" == function_name:
            function_value = self._linear(time_step, coefficients[0], coefficients[1])

        elif "constant" == function_name:
            function_value = self._constant(coefficients[0])

        elif "exponential" == function_name:
            function_value = self._exponential(time_step, coefficients[0])

        elif "logarithmic" == function_name:
            function_value = self._logarithmic(time_step, coefficients[0])

        elif "polynomial" in function_name:
            function_value = self._polynom(time_step, coefficients)

        return min(function_value, self.max_function_value) / norm_value

    def _calculate_derivative(self) -> np.array:
        """
        Calculate derivatives of each dimension, based on trajectories.

        Returns
        -------
        np.array
            derivatives for each dimension
        """
        # TODO: interval of derivatives, smooth derivative relative to action size and epochs?
        if self.c_step > 0:
            der = np.subtract(
                np.array(self.trajectory[self.c_step], dtype=np.float),
                np.array(self.trajectory[self.c_step - 1], dtype=np.float),
            )
        else:
            der = np.zeros(self.n_actions)

        return der

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

        # map integer action to vector
        action_vec = self.action_mapper[action]
        assert self.n_actions == len(
            action_vec
        ), f"action should be of length {self.n_actions}."
        self.action = action_vec

        # add trajectory and calculate derivatives
        # TODO: return action or action vector?
        self.trajectory.append(np.array(action_vec))
        self.trajectory_set[self.inst_id] = self.trajectory

        self.derivative = self._calculate_derivative()
        self.derivative_set[self.inst_id] = self.derivative

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

        self.trajectory = self.trajectory_set.get(
            self.inst_id, [np.zeros(self.n_actions)]
        )
        self.derivative = self.derivative_set.get(self.inst_id, 0)

        self._prev_state = None

        return self.get_state(self)

    def get_default_reward(self, _) -> float:
        """
        Calculate euclidean distance between action vector and real position of Curve.

        Parameters
        ----------
        _ : self
            ignore

        Returns
        -------
        float
            Euclidean distance
        """
        # get coordinates for all dimensions of the curve
        coordinates = np.zeros(self.n_actions)

        for dim, function_info in enumerate(self.instance):
            coordinates[dim] = self._calculate_function_value(
                self.c_step, function_info
            )

        # map action values to their interval mean
        mapping_list = list(self.action_interval_mapping.values())
        action_intervall = [
            mapping_list[count][index] for count, index in enumerate(self.action)
        ]

        # calculate euclidean norm
        dist = np.linalg.norm(action_intervall - coordinates)

        # norm reward to (0, 1)
        highest_coords = np.ones(max(self.action_vals))
        lowest_actions = np.full((max(self.action_vals)), np.min(mapping_list))
        max_dist = np.linalg.norm(highest_coords - lowest_actions)

        reward = 1 - (dist / max_dist)

        return reward

    def get_default_state(self, _) -> np.array:
        """
        Gather state information.

        Parameters
        ----------
        _ :
            ignore param

        Returns
        -------
        np.array
            numpy array with state information
        """
        remaining_budget = self.n_steps - self.c_step
        next_state = [remaining_budget]

        # append trajectories and derivatives and coordinates
        next_state.append(self.trajectory)
        next_state.append(self.derivative)

        # append multi-dim action vector
        if self.c_step == 0:
            next_state += [0 for _ in range(self.n_actions)]
        else:
            next_state += self.action

        return np.array(next_state)

    def close(self) -> bool:
        """
        Close Env

        Returns
        -------
        bool
            Closing confirmation
        """
        return True
