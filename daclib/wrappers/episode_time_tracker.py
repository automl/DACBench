import gym
from gym import Wrapper
import time


class EpisodeTimeWrapper(Wrapper):
    """
    Wrapper to track time spent per episode.
    Includes interval mode that return times in lists of len(interval) instead of one long list.
    """
    def __init__(self, env, config):
        super(EpisodeTimeWrapper, self).__init__(env)

        self.tracking_interval = config["tracking_interval"]
        self.overall = []
        if self.tracking_interval:
            self.interval_list = []
            self.current_interval = []

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

    def return_times(self):
        """
        Get times

        Returns
        -------
        np.array or np.array, np.array
            all times or all times and interval sorted times

        """
        if self.tracking_interval:
            np.array(self.overall), np.array(self.interval_list)
        else:
            return np.array(self.overall)

    # TODO!
    def render(self):
        return
