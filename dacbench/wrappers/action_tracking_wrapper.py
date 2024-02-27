"""Wrapper for action frequency."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from gymnasium import Wrapper, spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

sb.set_style("darkgrid")
current_palette = list(sb.color_palette())


class ActionFrequencyWrapper(Wrapper):
    """Wrapper to action frequency.

    Includes interval mode that returns frequencies in lists of len(interval)
    instead of one long list.
    """

    def __init__(self, env, action_interval=None, logger=None):
        """Initialize wrapper.

        Parameters
        ----------
        env : gym.Env
            Environment to wrap
        action_interval : int
            If not none, mean in given intervals is tracked, too
        logger: logger.ModuleLogger
            logger to write to

        """
        super().__init__(env)
        self.action_interval = action_interval
        self.overall_actions = []
        if self.action_interval:
            self.action_intervals = []
            self.current_actions = []
        self.action_space_type = type(self.env.action_space)
        self.logger = logger

    def __setattr__(self, name, value):
        """Set attribute in wrapper if available and in env if not.

        Parameters
        ----------
        name : str
            Attribute to set
        value
            Value to set attribute to

        """
        if name in [
            "action_interval",
            "overall_actions",
            "action_intervals",
            "current_actions",
            "env",
            "get_actions",
            "step",
            "render_action_tracking",
            "logger",
        ]:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def __getattribute__(self, name):
        """Get attribute value of wrapper if available and of env if not.

        Parameters
        ----------
        name : str
            Attribute to get

        Returns:
        -------
        value
            Value of given name

        """
        if name in [
            "action_interval",
            "overall_actions",
            "action_intervals",
            "current_actions",
            "env",
            "get_actions",
            "step",
            "render_action_tracking",
            "logger",
        ]:
            return object.__getattribute__(self, name)

        return getattr(self.env, name)

    def step(self, action):
        """Execute environment step and record state.

        Parameters
        ----------
        action : int
            action to execute

        Returns:
        -------
        np.array, float, bool, dict
            state, reward, done, metainfo

        """
        state, reward, terminated, truncated, info = self.env.step(action)
        self.overall_actions.append(action)
        if self.logger is not None:
            self.logger.log_space("action", action)

        if self.action_interval:
            if len(self.current_actions) < self.action_interval:
                self.current_actions.append(action)
            else:
                self.action_intervals.append(self.current_actions)
                self.current_actions = [action]
        return state, reward, terminated, truncated, info

    def get_actions(self):
        """Get state progression.

        Returns:
        -------
        np.array or np.array, np.array
            all states or all states and interval sorted states

        """
        if self.action_interval:
            complete_intervals = [*self.action_intervals, self.current_actions]
            return self.overall_actions, complete_intervals

        return self.overall_actions

    def render_action_tracking(self):
        """Render action progression.

        Returns:
        -------
        np.array
            RBG data of action tracking

        """

        def plot_single(ax=None, index=None, x=False, y=False):
            if ax is None:
                plt.xlabel("Step")
                plt.ylabel("Action value")
            elif x and y:
                ax.set_ylabel("Action value")
                ax.set_xlabel("Step")
            elif x:
                ax.set_xlabel("Step")
            elif y:
                ax.set_ylabel("Action value")

            if index is not None:
                ys = [state[index] for state in self.overall_actions]
            else:
                ys = self.overall_actions

            if ax is None:
                p = plt.plot(
                    np.arange(len(self.overall_actions)),
                    ys,
                    label="Step actions",
                    color="g",
                )
            else:
                p = ax.plot(
                    np.arange(len(self.overall_actions)),
                    ys,
                    label="Step actions",
                    color="g",
                )
            p2 = None
            if self.action_interval:
                if index is not None:
                    y_ints = []
                    for interval in self.action_intervals:
                        y_ints.append([state[index] for state in interval])
                else:
                    y_ints = self.action_intervals
                if ax is None:
                    p2 = plt.plot(
                        np.arange(len(self.action_intervals)) * self.action_interval,
                        [np.mean(interval) for interval in y_ints],
                        label="Mean interval action",
                        color="orange",
                    )
                    plt.legend(loc="upper left")
                else:
                    p2 = ax.plot(
                        np.arange(len(self.action_intervals)) * self.action_interval,
                        [np.mean(interval) for interval in y_ints],
                        label="Mean interval action",
                        color="orange",
                    )
                    ax.legend(loc="upper left")
            return p, p2

        action_size_border = 5
        if self.action_space_type == spaces.Discrete:
            figure = plt.figure(figsize=(12, 6))
            canvas = FigureCanvas(figure)
            p, p2 = plot_single()
            canvas.draw()
        elif self.action_space_type in (spaces.Dict, spaces.Tuple):
            raise NotImplementedError

        elif self.action_space_type in (
            spaces.MultiDiscrete,
            spaces.MultiBinary,
            spaces.Box,
        ):
            if self.action_space_type == spaces.MultiDiscrete:
                action_size = len(self.env.action_space.nvec)
            elif self.action_space_type == spaces.MultiBinary:
                action_size = self.env.action_space.n
            else:
                action_size = len(self.env.action_space.high)

            if action_size == 1:
                figure = plt.figure(figsize=(12, 6))
                canvas = FigureCanvas(figure)
                p, p2 = plot_single()
            elif action_size < action_size_border:
                dim = 1
                figure, axarr = plt.subplots(action_size)
            else:
                dim = action_size % 4
                figure, axarr = plt.subplots(action_size % 4, action_size // dim)
            figure.suptitle("State over time")
            canvas = FigureCanvas(figure)
            for i in range(action_size):
                if action_size == 1:
                    continue

                x = False
                if i % dim == dim - 1:
                    x = True
                if action_size < action_size_border:
                    p, p2 = plot_single(axarr[i], i, y=True, x=x)
                else:
                    y = i % action_size // dim == 0
                    p, p2 = plot_single(axarr[i % dim, i // dim], i, x=x, y=y)
            canvas.draw()
        width, height = figure.get_size_inches() * figure.get_dpi()
        return np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
