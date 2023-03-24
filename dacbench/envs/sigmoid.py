"""
Sigmoid environment from:

"Dynamic Algorithm Configuration:Foundation of a New Meta-Algorithmic Framework"
by A. Biedenkapp and H. F. Bozkurt and T. Eimer and F. Hutter and M. Lindauer.
Original environment authors: André Biedenkapp, H. Furkan Bozkurt
"""

from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from dacbench import AbstractMADACEnv


class SigmoidEnv(AbstractMADACEnv):
    """Environment for tracing sigmoid curves."""

    def _sig(self, x, scaling, inflection):
        """Simple sigmoid function"""
        return 1 / (1 + np.exp(-scaling * (x - inflection)))

    def __init__(self, config) -> None:
        """
        Initialize Sigmoid Env.

        Parameters
        ----------
        config : objdict
            Environment configuration

        """
        super(SigmoidEnv, self).__init__(config)

        self.shifts = [self.n_steps / 2 for _ in config["action_values"]]
        self.slopes = [-1 for _ in config["action_values"]]
        self.slope_multiplier = config["slope_multiplier"]
        self.n_actions = len(self.action_space.nvec)
        self._prev_state = None
        self.last_action = None

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
        Execute environment step.

        Parameters
        ----------
        action : int
            action to execute

        Returns
        -------
        np.array, float, bool, bool, dict
            state, reward, terminated, truncated, info

        """
        self.done = super(SigmoidEnv, self).step_()
        self.last_action = action
        next_state = self.get_state(self)
        self._prev_state = next_state
        return next_state, self.get_reward(self), False, self.done, {}

    def reset(self, seed=None, options={}) -> List[int]:
        """
        Resets env.

        Returns
        -------
        numpy.array
            Environment state

        """
        super(SigmoidEnv, self).reset_(seed)
        self.shifts = self.instance[: self.n_actions]
        self.slopes = self.instance[self.n_actions :]
        self._prev_state = None
        return self.get_state(self), {}

    def get_default_reward(self, _):
        """Get default reward."""
        r = [
            1 - np.abs(self._sig(self.c_step, slope, shift) - (act / (max_act - 1)))
            for slope, shift, act, max_act in zip(
                self.slopes, self.shifts, self.last_action, self.action_space.nvec
            )
        ]
        r = np.prod(r)
        r = max(self.reward_range[0], min(self.reward_range[1], r))
        return r

    def get_default_state(self, _):
        """Get default state representation."""
        remaining_budget = self.n_steps - self.c_step
        next_state = [remaining_budget]
        for shift, slope in zip(self.shifts, self.slopes):
            next_state.append(shift)
            next_state.append(slope)
        if self.c_step == 0:
            next_state += [-1 for _ in range(self.n_actions)]
        else:
            next_state = np.array(list(next_state) + list(self.last_action))
        return np.array(next_state)

    def close(self) -> bool:
        """
        Close Env.

        Returns
        -------
        bool
            Closing confirmation

        """
        return True

    def render(self, mode: str) -> None:
        """
        Render env in human mode.

        Parameters
        ----------
        mode : str
            Execution mode

        """
        if mode == "human" and self.n_actions == 2:
            plt.ion()
            plt.show()
            plt.cla()
            steps = np.arange(self.n_steps)
            self.data = self._sig(steps, self.slopes[0], self.shifts[0]) * self._sig(
                steps, self.slopes[1], self.shifts[1]
            ).reshape(-1, 1)

            plt.imshow(
                self.data,
                extent=(0, self.n_steps - 1, 0, self.n_steps - 1),
                interpolation="nearest",
                cmap=cm.plasma,
            )
            plt.axvline(x=self.c_step, color="r", linestyle="-", linewidth=2)
            plt.axhline(y=self.c_step, color="r", linestyle="-", linewidth=2)

            plt.draw()
            plt.pause(0.005)


class ContinuousStateSigmoidEnv(SigmoidEnv):
    """Environment for tracing sigmoid curves with a continuous state on the x-axis."""

    def __init__(self, config) -> None:
        """
        Initialize Sigmoid Env.

        Parameters
        ----------
        config : objdict
            Environment configuration

        """
        super().__init__(config)

    def step(self, action: int):
        """
        Execute environment step.

        Parameters
        ----------
        action : int
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, info

        """
        self.last_action = action
        # The reward measures how wrong the choice was so we can take this error to determine how far we travel along
        # the x-axis instead of always advancing + 1
        r = self.get_reward(self)

        # magic constants but such that the max step is ~1 and the min step is ~0.25
        self.c_step += (r + np.sqrt(np.power(r, 2) + 0.25)) / 2

        if self.c_step >= self.n_steps:
            self.done = True
        else:
            self.done = False

        # self.c_step is used in get_next_state to show how much distance along the x-axis is left to cover
        # Thus we get a continuous state this way.
        next_state = self.get_state(self)
        self._prev_state = next_state
        return next_state, r, self.done, {}


class ContinuousSigmoidEnv(SigmoidEnv):
    """Environment for tracing sigmoid curves with a continuous state on the x-axis."""

    def __init__(self, config) -> None:
        """
        Initialize Sigmoid Env.

        Parameters
        ----------
        config : objdict
            Environment configuration

        """
        super().__init__(config)

    def step(self, action: np.ndarray):
        """
        Execute environment step. !!NOTE!! The action here is a list of floats and not a single number !!NOTE!!

        Parameters
        ----------
        action : list of floats
            action(s) to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, info

        """
        self.last_action = action
        # The reward measures how wrong the choice was so we can take this error to determine how far we travel along
        # the x-axis instead of always advancing + 1
        r = self.get_reward(self)

        # magic constants but such that the max step is ~1 and the min step is ~0.25
        self.c_step += (r + np.sqrt(np.power(r, 2) + 0.25)) / 2

        if self.c_step >= self.n_steps:
            self.done = True
        else:
            self.done = False

        # self.c_step is used in get_next_state to show how much distance along the x-axis is left to cover
        # Thus we get a continuous state this way.
        next_state = self.get_state(self)
        self._prev_state = next_state
        return next_state, r, self.done, {}
