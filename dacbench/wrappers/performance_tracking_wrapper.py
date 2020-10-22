from gym import Wrapper
import numpy as np
from gym import spaces


class PerformanceTrackingWrapper(Wrapper):
    def __init__(self, env, performance_interval=None, track_instance_performance=True):
        super(PerformanceTrackingWrapper, self).__init__(env)
        self.performance_interval = performance_interval
        self.overall_performance = []
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
            "get_actions",
            "step",
            "render_action_tracking",
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
            "get_actions",
            "step",
            "render_action_tracking",
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
        self.overall_performance.append(reward)
        if self.performance_interval:
            if len(self.current_performance) < self.performance_interval:
                self.current_performance.append(reward)
            else:
                self.performance_intervals.append(self.current_performance)
                self.current_performance = [reward]
        if self.track_instances:
            if ''.join(self.env.instance) in self.instance_performances.keys():
                self.instance_performances[''.join(self.env.instance)].append(reward)
            else:
                self.instance_performances[''.join(self.env.instance)] = [reward]
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
            return self.overall_performance, complete_intervals, self.instance_performances
        elif self.performance_interval:
            complete_intervals = self.performance_intervals + [self.current_performance]
            return self.overall_performance, complete_intervals
        elif self.track_instances:
            return self.overall_performance, self.instance_performances
        else:
            return self.overall_performance
