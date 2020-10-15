from gym import Wrapper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time


class EpisodeTimeWrapper(Wrapper):
    """
    Wrapper to track time spent per episode.
    Includes interval mode that return times in lists of len(interval) instead of one long list.
    """

    def __init__(self, env, tracking_interval=None):
        super(EpisodeTimeWrapper, self).__init__(env)
        self.tracking_interval = tracking_interval
        self.overall = []
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
            "render_time_tracking"
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
            "render_time_tracking"
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
        self.overall.append(duration)
        if self.tracking_interval:
            if len(self.current_interval) < self.tracking_interval:
                self.current_interval.append(duration)
            else:
                self.interval_list.append(self.current_interval)
                self.current_interval = [duration]
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
            return self.overall, complete_intervals
        else:
            return np.array(self.overall)

    def render_time_tracking(self):
        """Render times"""
        figure = plt.figure(figsize=(12, 6))
        canvas = FigureCanvas(figure)
        plt.title("Time per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Time (s)")

        plt.plot(
            np.arange(len(self.overall)), self.overall, label="Episode time", color="b"
        )
        if self.tracking_interval:
            plt.plot(
                np.arange(len(self.interval_list)),
                [np.mean(interval) for interval in self.interval_list],
                label="Interval time",
                color="r",
            )

        canvas.draw()
        width, height = figure.get_size_inches() * figure.get_dpi()
        img = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
        return img
