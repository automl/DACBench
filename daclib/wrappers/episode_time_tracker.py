import gym
from gym import Wrapper

class EpisodeTimeWrapper(Wrapper):
    def __init__(self, config):
        super(EpisodeTimeWrapper, self).__init__(env)
        
        #TODO: separate discrete and continuous state components
        tracking_interval=config["tracking_interval"]

    def render(self):
        return
