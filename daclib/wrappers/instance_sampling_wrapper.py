import gym
from gym import Wrapper

class InstanceSamplingWrapper(Wrapper):
    def __init__(self, config):
        super(InstanceSamplingWrapper, self).__init__(env)
        if config['dist']:
            self.dist = config['dist']
        elif config['instance_path']:
            self.dist = fit_dist_from_file(instance_path)
        elif config['sampling_function']:
            self.sampling_function = config['sampling_function']
        else:
            print("No distribution to sample from given")
            return

    def sample(self):
        return

    def fit_dist_from_file(self, path):
        return
