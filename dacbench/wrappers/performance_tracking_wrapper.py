"""Wrapper for performance tracking."""
from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from gymnasium import Wrapper

sb.set_style("darkgrid")
current_palette = list(sb.color_palette())


class PerformanceTrackingWrapper(Wrapper):
    """Wrapper to track episode performance.

    Includes interval mode that returns performance in lists of len(interval)
    instead of one long list.
    """

    def __init__(
        self,
        env,
        performance_interval=None,
        track_instance_performance=True,
        logger=None,
    ):
        """Initialize wrapper.

        Parameters
        ----------
        env : gym.Env
            Environment to wrap
        performance_interval : int
            If not none, mean in given intervals is tracked, too
        track_instance_performance : bool
            Indicates whether to track per-instance performance
        logger : dacbench.logger.ModuleLogger
            logger to write to

        """
        super().__init__(env)
        self.performance_interval = performance_interval
        self.overall_performance = []
        self.episode_performance = 0
        if self.performance_interval:
            self.performance_intervals = []
            self.current_performance = []
        self.track_instances = track_instance_performance
        if self.track_instances:
            self.instance_performances = defaultdict(list)

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
            "performance_interval",
            "track_instances",
            "overall_performance",
            "performance_intervals",
            "current_performance",
            "env",
            "get_performance",
            "step",
            "instance_performances",
            "episode_performance",
            "render_performance",
            "render_instance_performance",
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
            "performance_interval",
            "track_instances",
            "overall_performance",
            "performance_intervals",
            "current_performance",
            "env",
            "get_performance",
            "step",
            "instance_performances",
            "episode_performance",
            "render_performance",
            "render_instance_performance",
            "logger",
        ]:
            return object.__getattribute__(self, name)

        return getattr(self.env, name)

    def step(self, action):
        """Execute environment step and record performance.

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
        self.episode_performance += reward

        if terminated or truncated:
            self.overall_performance.append(self.episode_performance)
            if self.logger is not None:
                self.logger.log(
                    "overall_performance",
                    self.episode_performance,
                )

            if self.performance_interval:
                if len(self.current_performance) < self.performance_interval:
                    self.current_performance.append(self.episode_performance)
                else:
                    self.performance_intervals.append(self.current_performance)
                    self.current_performance = [self.episode_performance]
            if self.track_instances:
                key = "".join(str(e) for e in self.env.instance)
                self.instance_performances[key].append(self.episode_performance)
            self.episode_performance = 0
        return state, reward, terminated, truncated, info

    def get_performance(self):
        """Get state performance.

        Returns:
        -------
        np.array or np.array, np.array or np.array, dict or np.array, np.arry, dict
            all states or all states and interval sorted states

        """
        if self.performance_interval and self.track_instances:
            complete_intervals = [*self.performance_intervals, self.current_performance]
            return (
                self.overall_performance,
                complete_intervals,
                self.instance_performances,
            )

        if self.performance_interval:
            complete_intervals = [*self.performance_intervals, self.current_performance]
            return self.overall_performance, complete_intervals

        if self.track_instances:
            return self.overall_performance, self.instance_performances

        return self.overall_performance

    def render_performance(self):
        """Plot performance."""
        plt.figure(figsize=(12, 6))
        plt.plot(
            np.arange(len(self.overall_performance) // 2),
            self.overall_performance[1::2],
        )
        plt.title("Mean Performance per episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()

    def render_instance_performance(self):
        """Plot mean performance for each instance."""
        plt.figure(figsize=(12, 6))
        plt.title("Mean Performance per Instance")
        plt.ylabel("Mean reward")
        plt.xlabel("Instance")
        ax = plt.subplot(111)
        for k, i in zip(
            self.instance_performances.keys(),
            np.arange(len(self.instance_performances.keys())),
            strict=False,
        ):
            ax.bar(str(i), np.mean(self.instance_performances[k]))
        plt.show()
