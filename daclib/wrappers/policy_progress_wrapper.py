import gym
from gym import Wrapper

class PolicyProgressWrapper(Wrapper):
    def __init__(self, env, config):
        super(PolicyProgressWrapper, self).__init__(env)
        #TODO: check for optimal policy in config

    def render(self):
        return
