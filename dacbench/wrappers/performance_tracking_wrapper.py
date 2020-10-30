from gym import Wrapper
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

sb.set_style("darkgrid")
current_palette = list(sb.color_palette())


class PerformanceTrackingWrapper(Wrapper):
    def __init__(self, env, performance_interval=None, track_instance_performance=True):
        super(PerformanceTrackingWrapper, self).__init__(env)
        self.performance_interval = performance_interval
        self.overall_performance = []
        self.episode_performance = 0
        if self.performance_interval:
            self.performance_intervals = []
            self.current_performance = []
        self.track_instances = track_instance_performance
        if self.track_instances:
            self.instance_performances = {}

    def __setattr__(self, name, value):
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
        ]:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def __getattribute__(self, name):
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
        ]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self.env, name)

    def step(self, action):
        """
        Execute environment step and record performance

        Parameters
        ----------
        action : int
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, metainfo
        """
        state, reward, done, info = self.env.step(action)
        self.episode_performance += reward
        if done:
            self.overall_performance.append(self.episode_performance)
            if self.performance_interval:
                if len(self.current_performance) < self.performance_interval:
                    self.current_performance.append(self.episode_performance)
                else:
                    self.performance_intervals.append(self.current_performance)
                    self.current_performance = [self.episode_performance]
            if self.track_instances:
                if (
                    "".join(str(e) for e in self.env.instance)
                    in self.instance_performances.keys()
                ):
                    self.instance_performances[
                        "".join(str(e) for e in self.env.instance)
                    ].append(self.episode_performance)
                else:
                    self.instance_performances[
                        "".join(str(e) for e in self.env.instance)
                    ] = [self.episode_performance]
            self.episode_performance = 0
        return state, reward, done, info

    def get_performance(self):
        """
        Get state performance

        Returns
        -------
        np.array or np.array, np.array or np.array, dict or np.array, np.arry, dict
            all states or all states and interval sorted states

        """
        if self.performance_interval and self.track_instances:
            complete_intervals = self.performance_intervals + [self.current_performance]
            return (
                self.overall_performance,
                complete_intervals,
                self.instance_performances,
            )
        elif self.performance_interval:
            complete_intervals = self.performance_intervals + [self.current_performance]
            return self.overall_performance, complete_intervals
        elif self.track_instances:
            return self.overall_performance, self.instance_performances
        else:
            return self.overall_performance

    def render_performance(self):
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
        plt.figure(figsize=(12, 6))
        plt.title("Mean Performance per Instance")
        plt.ylabel("Mean reward")
        plt.xlabel("Instance")
        ax = plt.subplot(111)
        for k, i in zip(
            self.instance_performances.keys(),
            np.arange(len(self.instance_performances.keys())),
        ):
            ax.bar(str(i), np.mean(self.instance_performances[k]))
        plt.show()
