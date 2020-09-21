import gym
from gym import Wrapper
from scipy.stats import norm

#TODO: implement this somewhere
import read_instances

class InstanceSamplingWrapper(Wrapper):
    def __init__(self, env, config):
        super(InstanceSamplingWrapper, self).__init__(env)
        if config['sampling_function']:
            self.sampling_function = config['sampling_function']
        elif config['path']:
            self.sampling_function = fit_dist_from_file(path)
        else:
            print("No distribution to sample from given")
            return

    def reset(self):
        instance = self.sampling_function()
        env.set_instance_set(instance)
        env.set_inst_id(0)
        return self.env.reset()

    #TODO: check if this actually works
    def fit_dist_from_file(self, path):
        instances = read_instances(path)
        dists = []
        for i in len(instances[0]):
            component = [inst[i] for inst in instances]
            dist = norm.fit(component)
            dists.append(dist)

        def sample():
            instance = []
            for d in dists:
                instance.append(d.rvs())
            return instance

        return sample
