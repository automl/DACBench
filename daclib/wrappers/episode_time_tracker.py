import gym
from gym import Wrapper
import time


class EpisodeTimeWrapper(Wrapper):
    def __init__(self, env, config):
        super(EpisodeTimeWrapper, self).__init__(env)

        self.tracking_interval = config["tracking_interval"]
        self.overall = []
        if self.tracking_interval:
            self.interval_list = []
            self.current_interval = []

    def step(self, action):
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

    def render(self):
        return
