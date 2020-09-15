import gym

class AbstractEnv(gym.Env):
    def __init__(self, config):
        self.super.__init__()
        self.instance_set = config['instance_set']
        self.inst_id = 0
        self.instance = self.instance_set[self.inst_id]
        return

    def reset(self):
        self.inst_id = (self.inst_id + 1) % len(self.instance_set)
        self.instance = self.instance_set[self.inst_id]
        return

    def get_inst_id(self):
        return self.inst_id

    def get_instance_set(self):
        return self.instance_set

    def get_instance(self):
        return self.instance
