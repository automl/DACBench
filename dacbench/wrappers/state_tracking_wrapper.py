"""Wrapper for the state tracking."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from gymnasium import Wrapper, spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

sb.set_style("darkgrid")
current_palette = list(sb.color_palette())


class StateTrackingWrapper(Wrapper):
    """Wrapper to track state changed over time.

    Includes interval mode that returns states in lists of len(interval)
    instead of one long list.
    """

    def __init__(self, env, state_interval=None, logger=None):
        """Initialize wrapper.

        Parameters
        ----------
        env : gym.Env
            Environment to wrap
        state_interval : int
            If not none, mean in given intervals is tracked, too
        logger : dacbench.logger.ModuleLogger
            logger to write to

        """
        super().__init__(env)
        self.state_interval = state_interval
        self.overall_states = []
        if self.state_interval:
            self.state_intervals = []
            self.current_states = []
        self.episode_states = None
        self.state_type = type(env.observation_space)
        self.logger = logger
        if self.logger is not None:
            benchmark_info = getattr(env, "benchmark_info", None)
            self.state_description = (
                benchmark_info.get("state_description", None)
                if benchmark_info is not None
                else None
            )

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
            "state_interval",
            "overall_states",
            "state_intervals",
            "current_states",
            "state_type",
            "env",
            "episode_states",
            "get_states",
            "step",
            "reset",
            "render_state_tracking",
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
            "state_interval",
            "overall_states",
            "state_intervals",
            "current_states",
            "state_type",
            "env",
            "episode_states",
            "get_states",
            "step",
            "reset",
            "render_state_tracking",
            "logger",
        ]:
            return object.__getattribute__(self, name)
        return getattr(self.env, name)

    def reset(self):
        """Reset environment and record starting state.

        Returns:
        -------
        np.array, {}
            state, info

        """
        state, info = self.env.reset()
        self.overall_states.append(state)
        if self.state_interval:
            if len(self.current_states) < self.state_interval:
                self.current_states.append(state)
            else:
                self.state_intervals.append(self.current_states)
                self.current_states = [state]
        return state, info

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
        self.overall_states.append(state)
        if self.logger is not None:
            self.logger.log_space("state", state, self.state_description)
        if self.state_interval:
            if len(self.current_states) < self.state_interval:
                self.current_states.append(state)
            else:
                self.state_intervals.append(self.current_states)
                self.current_states = [state]
        return state, reward, terminated, truncated, info

    def get_states(self):
        """Get state progression.

        Returns:
        -------
        np.array or np.array, np.array
            all states or all states and interval sorted states

        """
        if self.state_interval:
            complete_intervals = [*self.state_intervals, self.current_states]
            return self.overall_states, complete_intervals

        return self.overall_states

    def render_state_tracking(self):
        """Render state progression.

        Returns:
        -------
        np.array
            RBG data of state tracking

        """

        def plot_single(ax=None, index=None, x=False, y=False):
            if ax is None:
                ax = plt
                ax.xlabel("Episode")
                ax.ylabel("State")
            else:
                if x:
                    ax.set_xlabel("Episode")
                if y:
                    ax.set_ylabel("State")

            if index is not None:
                ys = [state[index] for state in self.overall_states]
            else:
                ys = self.overall_states

            p = ax.plot(
                np.arange(len(self.overall_states)),
                ys,
                label="Episode state",
                color="g",
            )

            p2 = None
            if self.state_interval:
                if index is not None:
                    y_ints = []
                    for interval in self.state_intervals:
                        y_ints.append([state[index] for state in interval])
                else:
                    y_ints = self.state_intervals

                p2 = ax.plot(
                    np.arange(len(self.state_intervals)) * self.state_interval,
                    [np.mean(interval) for interval in y_ints],
                    label="Mean interval state",
                    color="orange",
                )
                ax.legend(loc="upper left")

            return p, p2

        state_length_border = 5
        if self.state_type == spaces.Discrete:
            figure = plt.figure(figsize=(20, 20))
            canvas = FigureCanvas(figure)
            p, p2 = plot_single()
            canvas.draw()
        elif self.state_type in (spaces.Dict, spaces.Tuple):
            raise NotImplementedError

        elif self.state_type in (spaces.MultiDiscrete, spaces.MultiBinary, spaces.Box):
            if self.state_type == spaces.MultiDiscrete:
                state_length = len(self.env.observation_space.nvec)
            elif self.state_type == spaces.MultiBinary:
                state_length = self.env.observation_space.n
            else:
                state_length = len(self.env.observation_space.high)
            if state_length == 1:
                figure = plt.figure(figsize=(20, 20))
                canvas = FigureCanvas(figure)
                p, p2 = plot_single()
            elif state_length < state_length_border:
                dim = 1
                figure, axarr = plt.subplots(state_length)
            else:
                dim = state_length % 4
                figure, axarr = plt.subplots(state_length % 4, state_length // dim)
            figure.suptitle("State over time")
            canvas = FigureCanvas(figure)
            for i in range(state_length):
                if state_length == 1:
                    continue

                x = False
                if i % dim == dim - 1:
                    x = True
                if state_length < state_length_border:
                    p, p2 = plot_single(axarr[i], i, y=True, x=x)
                else:
                    y = i % state_length // dim == 0
                    p, p2 = plot_single(axarr[i % dim, i // dim], i, x=x, y=y)
            canvas.draw()
        else:
            raise ValueError("Unknown state type")
        width, height = figure.get_size_inches() * figure.get_dpi()
        return np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
