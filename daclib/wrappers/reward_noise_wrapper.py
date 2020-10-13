from gym import Wrapper


class RewardNoiseWrapper(Wrapper):
    def __init__(self, env, noise_function=None, noise_dist="gaussian"):
        super(RewardNoiseWrapper, self).__init__(env)

        if noise_function:
            self.noise_function = noise_function
        elif noise_dist == "gaussian":
            self.noise_function = self.add_gaussian()
        elif noise_dist == "exponential":
            self.noise_function = self.add_exponential()

        # self.noise_timing = config["noise_timing"]

    def __setattr__(self, name, value):
        if name in ["tracking_interval", "overall", "interval_list", "current_interval", "env"]:
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)

    def __getattr__(self, name):
        if name in ["tracking_interval", "overall", "interval_list", "current_interval", "env"]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self.env, name)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += self.noise_function()
        reward = max(self.env.reward_range[0], min(self.env.reward_range[1], reward))
        return state, reward, done, info

    def add_gaussian(self):
        return

    def add_exponential(self):
        return
