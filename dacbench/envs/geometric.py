"""
Geometric environment.
Original environment authors: Rasmus von Glahn
"""
import bisect
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from dacbench import AbstractEnv

sns.set_theme(style="darkgrid")


class GeometricEnv(AbstractEnv):
    """
    Environment for tracing different curves that are orthogonal to each other
    Use product approach: f(t,x,y,z) = X(t,x) * Y(t,y) * Z(t,z)
    Normalize Function Value on a Scale between 0 and 1
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
        self.realistic_trajectory = config["realistic_trajectory"]
        self.derivative_interval = config["derivative_interval"]

        self.correlation_table = config["correlation_table"]
        self.correlation_active = config["correlation_active"]
        self.correlation_depth = config["correlation_depth"]
        self.n_steps = config["cutoff"]

        self._prev_state = None
        self.action = None
        self.n_actions = len(self.action_vals)

        # Functions
        self.functions = Functions(
            self.n_steps,
            self.n_actions,
            len(self.instance_set),
            self.correlation_active,
            self.correlation_table,
            self.correlation_depth,
            self.derivative_interval,
        )
        self.functions.calculate_norm_values(self.instance_set)

        # Trajectories
        self.action_trajectory = []
        self.coord_trajectory = []
        self.action_trajectory_set = {}
        self.coord_trajectory_set = {}

        self.derivative = []
        self.derivative_set = {}

        if "reward_function" in config.keys():
            self.get_reward = config["reward_function"]
        else:
            self.get_reward = self.get_default_reward

        if "state_method" in config.keys():
            self.get_state = config["state_method"]
        else:
            self.get_state = self.get_default_state

    def get_optimal_policy(
        self, instance: List = None, vector_action: bool = True
    ) -> List[np.array]:
        """
        Calculates the optimal policy for an instance

        Parameters
        ----------
        instance : List, optional
            instance with information about function config.
        vector_action : bool, optional
            if True return multidim actions else return onedimensional action, by default True

        Returns
        -------
        List[np.array]
            List with entry for each timestep that holds all optimal values in an array or as int
        """
        if not instance:
            instance = self.instance

        optimal_policy_coords = self.functions.get_coordinates(instance).transpose()
        optimal_policy = np.zeros(((self.n_steps, self.n_actions)))

        for step in range(self.n_steps):
            for dimension in range(self.n_actions):
                step_size = 2 / self.action_vals[dimension]
                interval = [step for step in np.arange(-1, 1, step_size)][1:]

                optimal_policy[step, dimension] = bisect.bisect_left(
                    interval, optimal_policy_coords[step, dimension]
                )

        optimal_policy = optimal_policy.astype(int)
        return optimal_policy

    def step(self, action: int):
        """
        Execute environment step

        Parameters
        ----------
        action : int
            action to execute

        Returns
        -------
        np.array, float, bool, bool, dict
            state, reward, terminated, truncated, info
        """
        self.done = super(GeometricEnv, self).step_()
        self.action = action

        coords = self.functions.get_coordinates_at_time_step(self.c_step)
        self.coord_trajectory.append(coords)
        self.action_trajectory.append(action)

        self.coord_trajectory_set[self.inst_id] = self.coord_trajectory
        self.action_trajectory_set[self.inst_id] = self.action_trajectory

        if self.realistic_trajectory:
            self.derivative = self.functions.calculate_derivative(
                self.coord_trajectory, self.c_step
            )
        else:
            self.derivative = self.functions.calculate_derivative(
                self.action_trajectory, self.c_step
            )

        self.derivative_set[self.inst_id] = self.derivative

        next_state = self.get_state(self)
        self._prev_state = next_state

        reward = self.get_reward(self)
        if reward > 1:
            print(f"Instance: {self.instance}, Reward:{reward}, step: {self.c_step}")
            raise ValueError(f"Reward zu Hoch Coords: {coords}, step: {self.c_step}")
        if math.isnan(reward):
            raise ValueError(f"Reward NAN Coords: {coords}, step: {self.c_step}")

        return next_state, reward, False, self.done, {}

    def reset(self, seed=None, options={}) -> List[int]:
        """
        Resets env

        Returns
        -------
        numpy.array
            Environment state
        dict
            Meta-info
        """
        super(GeometricEnv, self).reset_(seed)
        self.functions.set_instance(self.instance, self.instance_index)

        if self.c_step:
            self.action_trajectory = self.action_trajectory_set.get(self.inst_id)

        self.coord_trajectory = self.coord_trajectory_set.get(
            self.inst_id, [self.functions.get_coordinates_at_time_step(self.c_step)]
        )

        self.derivative = self.derivative_set.get(
            self.inst_id, np.zeros(self.n_actions)
        )
        self._prev_state = None

        return self.get_state(self), {}

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
        coords, action_coords, highest_coords, lowest_actions = self._pre_reward()
        euclidean_dist = np.linalg.norm(action_coords - coords)

        max_dist = np.linalg.norm(highest_coords - lowest_actions)
        reward = 1 - (euclidean_dist / max_dist)

        return abs(reward)

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

        next_state += [self.n_actions]

        if self.c_step == 0:
            next_state += [0 for _ in range(self.n_actions)]
            next_state += [0 for _ in range(self.n_actions)]
        else:
            next_state += list(self.derivative)
            next_state += list(self.coord_trajectory[self.c_step])

        return np.array(next_state, dtype="float32")

    def close(self) -> bool:
        """
        Close Env

        Returns
        -------
        bool
            Closing confirmation
        """
        return True

    def render(self, dimensions: List, absolute_path: str):
        """
        Multiplot for specific dimensions of benchmark with policy actions.

        Parameters
        ----------
        dimensions : List
            List of dimensions that get plotted
        """
        coordinates = self.functions.get_coordinates()

        fig, axes = plt.subplots(
            len(dimensions), sharex=True, sharey=True, figsize=(15, 4 * len(dimensions))
        )
        plt.xlabel("time steps", fontsize=32)
        plt.ylim(-1.1, 1.1)
        plt.xlim(-0.1, self.n_steps - 0.9)
        plt.xticks(np.arange(0, self.n_steps, 1), fontsize=24.0)

        for idx, dim in zip(range(len(dimensions)), dimensions):
            function_info = self.instance[dim]
            title = function_info[1] + " - Dimension " + str(dim)
            axes[idx].tick_params(axis="both", which="major", labelsize=24)
            axes[idx].set_yticks((np.arange(-1, 1.1, 2 / self.action_vals[dim])))
            axes[idx].set_title(title, size=32)
            axes[idx].plot(coordinates[dim], label="Function", marker="o", linewidth=3)[
                0
            ].axes
            axes[idx].xaxis.grid(False)
            axes[idx].vlines(x=[3.5, 7.5], ymin=-1, ymax=1, colors="white", ls="--")
            """
            axes[idx].legend(
                loc="lower right",
                framealpha=1,
                shadow=True,
                borderpad=1,
                frameon=True,
                ncol=1,
                edgecolor="0.2",
            )
            """

        fig_title = f"GeoBench-Dimensions{len(dimensions)}"
        fig.savefig(os.path.join(absolute_path, fig_title + ".jpg"))

    def render_3d_dimensions(self, dimensions: List, absolute_path: str):
        """
        Plot 2 Dimensions in 3D space

        Parameters
        ----------
        dimensions : List
            List of dimensions that get plotted. Max 2
        """
        assert len(dimensions) == 2
        print(mplot3d)

        coordinates = self.functions.get_coordinates()

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")

        x = list(range(self.n_steps))
        z = coordinates[dimensions[0]][x]
        y = coordinates[dimensions[1]][x]

        ax.set_title("3D line plot")

        ax.plot3D(x, y, z, "blue")
        ax.view_init()
        fig.savefig(os.path.join(absolute_path, "3D.jpg"))

        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.view_init(elev=0, azim=-90)
        fig.savefig(os.path.join(absolute_path, "3D-90side.jpg"))

    def _pre_reward(self) -> Tuple[np.ndarray, List]:
        """
        Prepare actions and coordinates for reward calculation.

        Returns
        -------
        Tuple[np.ndarray, List]
            [description]
        """
        coordinates = self.functions.get_coordinates_at_time_step(self.c_step)
        function_names = [function_info[1] for function_info in self.instance]

        # map action values to their interval mean
        mapping_list = [self.action_interval_mapping[name] for name in function_names]
        action_intervall = [
            mapping_list[count][index] for count, index in enumerate(self.action)
        ]
        highest_coords = np.ones(self.n_actions)
        lowest_actions = np.array([val[0] for val in mapping_list])

        return coordinates, action_intervall, highest_coords, lowest_actions


class Functions:
    def __init__(
        self,
        n_steps: int,
        n_actions: int,
        n_instances: int,
        correlation: bool,
        correlation_table: np.ndarray,
        correlation_depth: int,
        derivative_interval: int,
    ) -> None:
        self.instance = None
        self.instance_idx = None

        self.coord_array = np.zeros((n_actions, n_steps))
        self.calculated_instance = None
        self.norm_calculated = False
        self.norm_values = np.ones((n_instances, n_actions))

        self.correlation = correlation
        self.correlation_table = correlation_table
        self.correlation_changes = np.zeros(n_actions)
        self.correlation_depth = correlation_depth

        self.n_steps = n_steps
        self.n_actions = n_actions
        self.derivative_interval = derivative_interval

    def set_instance(self, instance: List, instance_index):
        """update instance"""
        self.instance = instance
        self.instance_idx = instance_index

    def get_coordinates(self, instance: List = None) -> List[np.array]:
        """
        Calculates coordinates for instance over all time_steps.
        The values will change if correlation is applied and not optimal actions are taken.

        Parameters
        ----------
        instance : List, optional
            Instance that holds information about functions, by default None

        Returns
        -------
        List[np.array]
            Index of List refers to time step
        """
        if not instance:
            instance = self.instance
        assert instance

        if self.instance_idx == self.calculated_instance:
            optimal_coords = self.coord_array
        else:
            optimal_coords = np.zeros((self.n_actions, self.n_steps))
            for time_step in range(self.n_steps):
                optimal_coords[:, time_step] = self.get_coordinates_at_time_step(
                    time_step + 1
                )

            if self.norm_calculated:
                self.coord_array = optimal_coords
                self.calculated_instance = self.instance_idx

        return optimal_coords

    def get_coordinates_at_time_step(self, time_step: int) -> np.array:
        """
        Calculate coordiantes at time_step.
        Apply correlation.

        Parameters
        ----------
        instance : List
            Instance that holds information about functions
        time_step : int
            Time step of functions

        Returns
        -------
        np.array
            array of function values at timestep
        """
        if self.instance_idx == self.calculated_instance:
            value_array = self.coord_array[:, time_step - 1]
        else:
            value_array = np.zeros(self.n_actions)
            for index, function_info in enumerate(self.instance):
                value_array[index] = self._calculate_function_value(
                    time_step, function_info, index
                )

            if self.correlation and time_step > 1 and self.norm_calculated:
                value_array = self._add_correlation(value_array, time_step)

        return value_array

    def calculate_derivative(self, trajectory: List, c_step: int) -> np.array:
        """
        Calculate derivatives of each dimension, based on trajectories.

        Parameters
        ----------
        trajectory: List
            List of actions or coordinates already taken
        c_step: int
            current timestep

        Returns
        -------
        np.array
            derivatives for each dimension
        """
        if c_step > 1:
            upper_bound = c_step + 1
            lower_bound = max(upper_bound - self.derivative_interval, 1)

            derrivative = np.zeros(self.n_actions)
            for step in range(lower_bound, upper_bound):
                der = np.subtract(
                    np.array(trajectory[step], dtype=np.float32),
                    np.array(trajectory[step - 1], dtype=np.float32),
                )
                derrivative = np.add(derrivative, der)

            derrivative /= upper_bound - lower_bound

        elif c_step == 1:
            derrivative = np.subtract(
                np.array(trajectory[c_step], dtype=np.float32),
                np.array(trajectory[c_step - 1], dtype=np.float32),
            )

        else:
            derrivative = np.zeros(self.n_actions)

        return derrivative

    def calculate_norm_values(self, instance_set: Dict):
        """
        Norm Functions to Intervall between -1 and 1
        """
        for key, instance in instance_set.items():
            self.set_instance(instance, key)
            instance_values = self.get_coordinates()

            for dim, function_values in enumerate(instance_values):
                if abs(min(function_values)) > max(function_values):
                    norm_factor = abs(min(function_values))
                else:
                    norm_factor = max(function_values)

                self.norm_values[key][dim] = norm_factor

        self.norm_calculated = True

    def _calculate_function_value(
        self, time_step: int, function_infos: List, func_idx: int
    ) -> float:
        """
        Call different functions with their speicifc parameters and norm them.

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
        assert self.instance_idx == function_infos[0]

        function_name = function_infos[1]
        coefficients = function_infos[2:]
        if self.norm_calculated:
            norm_value = self.norm_values[self.instance_idx, func_idx]
            if norm_value == 0:
                norm_value = 1
        else:
            norm_value = 1

        function_value = 0

        if "sigmoid" == function_name:
            function_value = self._sigmoid(time_step, coefficients[0], coefficients[1])

        elif "linear" == function_name:
            function_value = self._linear(time_step, coefficients[0], coefficients[1])

        elif "constant" == function_name:
            function_value = self._constant(coefficients[0])

        elif "logarithmic" == function_name:
            function_value = self._logarithmic(time_step, coefficients[0])

        elif "cubic" in function_name:
            function_value = self._cubic(
                time_step, coefficients[0], coefficients[1], coefficients[2]
            )

        elif "parabel" in function_name:
            function_value = self._parabel(
                time_step, coefficients[0], coefficients[1], coefficients[2]
            )

        elif "sinus" in function_name:
            function_value = self._sinus(time_step, coefficients[0])

        function_value = np.round(function_value / norm_value, 5)
        if self.norm_calculated:
            function_value = max(min(function_value, 1), -1)

        return function_value

    def _add_correlation(self, value_array: np.ndarray, time_step: int):
        """
        Adds correlation between dimensions but clips at -1 and 1.
        Correlation table holds numbers between -1 and 1.
        e.g. correlation_table[0][2] = 0.5 if dimension 1 changes dimension 3 changes about 50% of dimension one

        Parameters
        ----------
        correlation_table : np.array
            table that holds all values of correlation between dimensions [n,n]
        """
        prev_values = self.coord_array[:, time_step - 1]
        diff_values = value_array - prev_values

        new_values = []
        for idx, diff in enumerate(diff_values):
            self._apply_correlation_update(idx, diff, self.correlation_depth)

        new_values = self.correlation_changes + value_array
        clipped_values = np.clip(new_values, a_min=-1, a_max=1)
        self.correlation_changes = np.zeros(self.n_actions)

        return clipped_values

    def _apply_correlation_update(self, idx: int, diff: float, depth):
        """
        Recursive function for correlation updates
        Call function recursively till depth is 0 or diff is too small.
        """
        if not depth or diff < 0.001:
            return

        for coeff_idx, corr_coeff in enumerate(self.correlation_table[:][idx]):
            change = corr_coeff * diff
            self.correlation_changes[coeff_idx] += change
            self._apply_correlation_update(coeff_idx, change, depth - 1)

    def _sigmoid(self, t: float, scaling: float, inflection: float):
        """Simple sigmoid function"""
        return 1 / (1 + np.exp(-scaling * (t - inflection)))

    def _linear(self, t: float, a: float, b: float):
        """Linear function"""
        return a * t + b

    def _parabel(self, t: float, sig: int, x_int: int, y_int: int):
        """Parabel function"""
        return sig * (t - x_int) ** 2 + y_int

    def _cubic(self, t: float, sig: int, x_int: int, y_int: int):
        """cubic function"""
        return sig * (t - x_int) ** 3 + y_int

    def _logarithmic(self, t: float, a: float):
        """Logarithmic function"""
        if t != 0:
            return a * np.log(t)
        else:
            return 1000

    def _constant(self, c: float):
        """Constant function"""
        return c

    def _sinus(self, t: float, scale: float):
        """Sinus function"""
        return np.sin(scale * t)
