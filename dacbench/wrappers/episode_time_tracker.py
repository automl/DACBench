from gym import Wrapper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
import seaborn as sb

# sb.set_style("darkgrid")
current_palette = list(sb.color_palette())


class EpisodeTimeWrapper(Wrapper):
    """
    Wrapper to track time spent per episode.
    Includes interval mode that return times in lists of len(interval) instead of one long list.
    """

    def __init__(self, env, tracking_interval=None):
        super(EpisodeTimeWrapper, self).__init__(env)
        self.tracking_interval = tracking_interval
        self.all_steps = []
        if self.tracking_interval:
            self.step_intervals = []
            self.current_step_interval = []
        self.overall = []
        self.episode = []
        if self.tracking_interval:
            self.interval_list = []
            self.current_interval = []

    def __setattr__(self, name, value):
        if name in [
            "tracking_interval",
            "overall",
            "interval_list",
            "current_interval",
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
        ]:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def __getattribute__(self, name):
        if name in [
            "tracking_interval",
            "overall",
            "interval_list",
            "current_interval",
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
        ]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self.env, name)

    def step(self, action):
        """
        Execute environment step and record time

        Parameters
        ----------
        action : int
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, metainfo
        """
        start = time.time()
        state, reward, done, info = self.env.step(action)
        stop = time.time()
        duration = stop - start
        self.episode.append(duration)
        self.all_steps.append(duration)
        if self.tracking_interval:
            if len(self.current_step_interval) < self.tracking_interval:
                self.current_step_interval.append(duration)
            else:
                self.step_intervals.append(self.current_step_interval)
                self.current_step_interval = [duration]
        if done:
            self.overall.append(self.episode)
            if self.tracking_interval:
                if len(self.current_interval) < self.tracking_interval:
                    self.current_interval.append(self.episode)
                else:
                    self.interval_list.append(self.current_interval)
                    self.current_interval = []
            self.episode = []
        return state, reward, done, info

    def get_times(self):
        """
        Get times

        Returns
        -------
        np.array or np.array, np.array
            all times or all times and interval sorted times

        """
        if self.tracking_interval:
            complete_intervals = self.interval_list + [self.current_interval]
            complete_step_intervals = self.step_intervals + [self.current_step_interval]
            return (
                self.overall,
                self.all_steps,
                complete_intervals,
                complete_step_intervals,
            )
        else:
            return np.array(self.overall), np.array(self.all_steps)

    def render_step_time(self):
        """Render step times"""
        figure = plt.figure(figsize=(12, 6))
        canvas = FigureCanvas(figure)
        plt.title("Time per Step")
        plt.xlabel("Step")
        plt.ylabel("Time (s)")

        plt.plot(
            np.arange(len(self.all_steps)),
            self.all_steps,
            label="Step time",
            color="g",
        )
        if self.tracking_interval:
            interval_means = [np.mean(interval) for interval in self.step_intervals] + [
                np.mean(self.current_step_interval)
            ]
            plt.plot(
                np.arange(len(self.step_intervals) + 2) * self.tracking_interval,
                [interval_means[0]] + interval_means,
                label="Mean interval time",
                color="orange",
            )
        plt.legend(loc="upper right")
        canvas.draw()
        width, height = figure.get_size_inches() * figure.get_dpi()
        img = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
        # plt.close(figure)
        return img

    def render_episode_time(self):
        """Render episode times"""
        figure = plt.figure(figsize=(12, 6))
        canvas = FigureCanvas(figure)
        plt.title("Time per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Time (s)")

        plt.plot(
            np.arange(len(self.overall)),
            [sum(episode) for episode in self.overall],
            label="Episode time",
            color="g",
        )
        if self.tracking_interval:
            interval_sums = []
            for interval in self.interval_list:
                ep_times = []
                for episode in interval:
                    ep_times.append(sum(episode))
                interval_sums.append(np.mean(ep_times))
            interval_sums += [
                np.mean([sum(episode) for episode in self.current_interval])
            ]
            plt.plot(
                np.arange(len(self.interval_list) + 2) * self.tracking_interval,
                [interval_sums[0]] + interval_sums,
                label="Mean interval time",
                color="orange",
            )
        plt.legend(loc="upper right")
        canvas.draw()
        width, height = figure.get_size_inches() * figure.get_dpi()
        img = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
        return img
