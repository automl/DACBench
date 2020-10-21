from gym import Wrapper
import numpy as np


class RewardNoiseWrapper(Wrapper):
    def __init__(
        self, env, noise_function=None, noise_dist="standard_normal", dist_args=None
    ):
        super(RewardNoiseWrapper, self).__init__(env)

        if noise_function:
            self.noise_function = noise_function
        elif noise_dist:
            self.noise_function = self.add_noise(noise_dist, dist_args)
        else:
            raise Exception("No distribution to sample noise from given")

    def __setattr__(self, name, value):
        if name in ["noise_function", "env", "add_noise", "step"]:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def __getattribute__(self, name):
        if name in ["noise_function", "env", "add_noise", "step"]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self.env, name)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        print(reward)
        reward += self.noise_function()
        reward = max(self.env.reward_range[0], min(self.env.reward_range[1], reward))
        return state, reward, done, info

    def add_noise(self, dist, args):
        rng = np.random.default_rng()
        function = getattr(rng, dist)

        def sample_noise():
            if args:
                return function(*args)
            else:
                return function()

        return sample_noise
