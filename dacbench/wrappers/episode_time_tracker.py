"""Wrapper to track time."""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from gymnasium import Wrapper
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

sb.set_style("darkgrid")
current_palette = list(sb.color_palette())


class EpisodeTimeWrapper(Wrapper):
    """Wrapper to track time spent per episode.

    Includes interval mode that returns times in lists of len(interval)
    instead of one long list.
    """

    def __init__(self, env, time_interval=None, logger=None):
        """Initialize wrapper.

        Parameters
        ----------
        env : gym.Env
            Environment to wrap
        time_interval : int
            If not none, mean in given intervals is tracked, too
        logger : dacbench.logger.ModuleLogger
            logger to write to

        """
        super().__init__(env)
        self.time_interval = time_interval
        self.all_steps = []
        if self.time_interval:
            self.step_intervals = []
            self.current_step_interval = []
        self.overall_times = []
        self.episode = []
        if self.time_interval:
            self.time_intervals = []
            self.current_times = []

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
            "time_interval",
            "overall_times",
            "time_intervals",
            "current_times",
            "env",
            "get_times",
            "step",
            "render_step_time",
            "render_episode_time",
            "reset",
            "episode",
            "all_steps",
            "current_step_interval",
            "step_intervals",
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
            "time_interval",
            "overall_times",
            "time_intervals",
            "current_times",
            "env",
            "get_times",
            "step",
            "render_step_time",
            "render_episode_time",
            "reset",
            "episode",
            "all_steps",
            "current_step_interval",
            "step_intervals",
            "logger",
        ]:
            return object.__getattribute__(self, name)

        return getattr(self.env, name)

    def step(self, action):
        """Execute environment step and record time.

        Parameters
        ----------
        action : int
            action to execute

        Returns:
        -------
        np.array, float, bool, bool, dict
            state, reward, terminated, truncated, metainfo

        """
        start = time.time()
        state, reward, terminated, truncated, info = self.env.step(action)
        stop = time.time()
        duration = stop - start
        self.episode.append(duration)
        self.all_steps.append(duration)

        if self.logger is not None:
            self.logger.log("step_duration", duration)

        if self.time_interval:
            if len(self.current_step_interval) < self.time_interval:
                self.current_step_interval.append(duration)
            else:
                self.step_intervals.append(self.current_step_interval)
                self.current_step_interval = [duration]
        if terminated or truncated:
            self.overall_times.append(self.episode)
            if self.logger is not None:
                self.logger.log("episode_duration", sum(self.episode))

            if self.time_interval:
                if len(self.current_times) < self.time_interval:
                    self.current_times.append(self.episode)
                else:
                    self.time_intervals.append(self.current_times)
                    self.current_times = []
            self.episode = []
        return state, reward, terminated, truncated, info

    def get_times(self):
        """Get times.

        Returns:
        -------
        np.array or np.array, np.array
            all times or all times and interval sorted times

        """
        if self.time_interval:
            complete_intervals = [*self.time_intervals, self.current_times]
            complete_step_intervals = [*self.step_intervals, self.current_step_interval]
            return (
                self.overall_times,
                self.all_steps,
                complete_intervals,
                complete_step_intervals,
            )

        return np.array(self.overall_times), np.array(self.all_steps)

    def render_step_time(self):
        """Render step times."""
        figure = plt.figure(figsize=(12, 6))
        canvas = FigureCanvas(figure)
        plt.title("Time per Step")
        plt.xlabel("Step")
        plt.ylabel("Time (s)")

        plt.plot(
            np.arange(len(self.all_steps)), self.all_steps, label="Step time", color="g"
        )
        if self.time_interval:
            interval_means = [np.mean(interval) for interval in self.step_intervals] + [
                np.mean(self.current_step_interval)
            ]
            plt.plot(
                np.arange(len(self.step_intervals) + 2) * self.time_interval,
                [interval_means[0], *interval_means],
                label="Mean interval time",
                color="orange",
            )
        plt.legend(loc="upper right")
        canvas.draw()
        width, height = figure.get_size_inches() * figure.get_dpi()
        return np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
        # plt.close(figure)

    def render_episode_time(self):
        """Render episode times."""
        figure = plt.figure(figsize=(12, 6))
        canvas = FigureCanvas(figure)
        plt.title("Time per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Time (s)")

        plt.plot(
            np.arange(len(self.overall_times)),
            [sum(episode) for episode in self.overall_times],
            label="Episode time",
            color="g",
        )
        if self.time_interval:
            interval_sums = []
            for interval in self.time_intervals:
                ep_times = []
                for episode in interval:
                    ep_times.append(sum(episode))
                interval_sums.append(np.mean(ep_times))
            interval_sums += [np.mean([sum(episode) for episode in self.current_times])]
            plt.plot(
                np.arange(len(self.time_intervals) + 2) * self.time_interval,
                [interval_sums[0], *interval_sums],
                label="Mean interval time",
                color="orange",
            )
        plt.legend(loc="upper right")
        canvas.draw()
        width, height = figure.get_size_inches() * figure.get_dpi()
        return np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
