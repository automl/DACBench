import gym
from gym import Wrapper


class RewardNoiseWrapper(Wrapper):
    def __init__(self, env, config):
        super(RewardNoiseWrapper, self).__init__(env)

        # TODO: get dimension and other info on reward
        if "noise_function" in config.keys():
            self.noise_function = config["noise_function"]
        elif config["noise_dist"] == "gaussian":
            self.noise_function = self.add_gaussian()
        elif config["noise_dist"] == "exponential":
            self.noise_function = self.add_exponential()

        #self.noise_timing = config["noise_timing"]

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += self.noise_function()
        reward = max(self.env.reward_range[0], min(self.env.reward_range[1], reward))
        return state, reward, done, info

    def add_gaussian(self):
        return

    def add_exponential(self):
        return
