import gym
from gym import Wrapper


class RewardNoiseWrapper(Wrapper):
    def __init__(self, env, config):
        super(RewardNoiseWrapper, self).__init__(env)

        # TODO: get dimension and other info on reward
        noise_dist = config["noise_dist"]
        noise_timing = config["noise_timing"]

    def add_gaussian(self):
        return

    def add_exponential(self):
        return
